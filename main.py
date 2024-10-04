import os
import torch
import transformers

from rotatellm.quant_utils import *
from rotatellm.rotation_utils import *
from rotatellm.gptq_utils import *
from utils.eval_utils import *
from utils.hadamard_utils import *
from utils.model_utils import *
from utils.data_utils import *
from utils.utils import *


def main():
    args, logger = parser_gen()
    
    transformers.set_seed(args.seed)
    model = get_model(args.model, args.hf_token)
    model.eval()
    
    # Rotate the weights
    if args.rotate:
        logger.info("====== Rotate ======")
        fuse_layer_norms(model)
        rotate_model(model, args, logger)
        cleanup_memory(logger, verbos=True)
            
        add_actquant(model) #Add Activation Wrapper to the model
        qlayers = find_qlayers(model)
        for name in qlayers:
            if 'down_proj' in name:
                had_K, K = get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if 'o_proj' in name:
                had_K, K = get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
    else:
        add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present
        

    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path: # Load Quantized Rotated Model
            logger.info("===== Load Quantized Model =====")

            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            et = args.model.split("/")[-1]
            file_name = f"{net}_rotate_w{args.w_bits}_a{args.a_bits}_v{args.v_bits}_k{args.k_bits}.pth"
            # 完整的文件路径
            full_path = os.path.join(args.save_qmodel_path, file_name)
            print("Load quantized model from ", full_path)
            save_dict = torch.load(full_path)
            model.load_state_dict(save_dict["model"])
            
        elif not args.w_rtn: # GPTQ Weight Quantization
            logger.info("===== GPTQ Weight Quantization =====")
            assert "llama" in args.model, "Only llama is supported for GPTQ!"
            
            trainloader = get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            quantizers = gptq_fwrd(model, trainloader, DEV, args, logger)
            save_dict["w_quantizers"] = quantizers
        else: # RTN Weight Quantization
            logger.info("===== RTN Weight Quantization =====")
            quantizers = rtn_fwrd(model, DEV, args)
            save_dict["w_quantizers"] = quantizers
            
        if args.save_qmodel_path:
            logger.info("===== Save Quantization Model =====")
            # 确保目录存在
            if not os.path.exists(args.save_qmodel_path):
                os.makedirs(args.save_qmodel_path)
            net = args.model.split("/")[-1]
            file_name = f"{net}_rotate_w{args.w_bits}_a{args.a_bits}_v{args.v_bits}_k{args.k_bits}.pth"
            # 完整的文件路径
            full_path = os.path.join(args.save_qmodel_path, file_name)
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, full_path)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        logger.info("===== Start Quantization =====")

        qlayers = find_qlayers(model, layers=[ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and "llama" in args.model:
            down_proj_groupsize = llama_down_proj_groupsize(model, args.a_groupsize, logger)
        
        for name in qlayers:            
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not(args.a_asym)
            layer_a_clip = args.a_clip_ratio
            
            if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                              groupsize=args.v_groupsize,
                                              sym=not(args.v_asym),
                                              clip_ratio=args.v_clip_ratio)
            
            if 'lm_head' in name: #Skip lm_head quantization   
                layer_input_bits = 16
            
            if 'down_proj' in name: #Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize
                
            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                              groupsize=layer_groupsize,
                                              sym=layer_a_sym,
                                              clip_ratio=layer_a_clip)


    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = get_rope_function_name(model)  # apply_rotary_pos_emb
            layers = get_layers(model)
            k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
            for layer in layers:
                add_qk_rotation_wrapper_after_function_call_in_forward(
                            layer.self_attn, 
                            rope_function_name, 
                            config=model.config,
                            **k_quant_config)
        
    # Evaluating on dataset
    testloader = get_loaders(
            args.eval_dataset,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True
        )

    
    dataset_ppl = evaluator(model, testloader, DEV, args)

    if not args.lm_eval:
        return
    else:
        # Import lm_eval utils
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.models.huggingface import HFLM

    
    if args.distribute:
        distribute_model(model)
    else:
        model.to(DEV)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

    task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)
    results = lm_eval.simple_evaluate(hflm, tasks=task_names, batch_size=args.lm_eval_batch_size)['results']

    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    print(metric_vals)

if __name__ == '__main__':
    main()
