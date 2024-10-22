import os
import torch
import torch.distributed as dist
import transformers

from utils.eval_utils import *
from utils.hadamard_utils import *
from utils.model_utils import *
from utils.data_utils import *
from utils.utils import *
from utils.fuse_norm_utils import *
from utils.quant_utils import *
from utils.process_args import parser_gen


def get_llama(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                          use_auth_token=hf_token,
                                                          device_map='auto',
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def main():
    args, logger = parser_gen()
    
    transformers.set_seed(args.seed)
    model = get_llama(args.model, args.hf_token)
    
    if args.methods == "quarot":
        from train_utils.main import quarot_train

        # Rotate the weights
        model = quarot_train(model, args, logger)
            
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

    elif args.methods == "spinquant":
        from train_utils.main import spinquant_train
        model = spinquant_train(model, args, logger)

        for param in model.parameters():
            param.requires_grad = False
        R1 = random_hadamard_matrix(model.config.hidden_size, "cuda")
        model.R1 = RotateModule(R1)
        for i in range(model.config.num_hidden_layers):
            # Each head dim = 128 for Llama model
            R2 = random_hadamard_matrix(
                model.config.hidden_size // model.config.num_attention_heads, "cuda"
            )
            model.model.layers[i].self_attn.R2 = RotateModule(R2)
        if local_rank == 0:
            logger.info("Model init completed for training {}".format(model))
            logger.info("Start to load tokenizer...")
        tokenizer = LlamaTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=args.model,
            padding_side="right",
            use_fast=True,
            add_eos_token=False,
            add_bos_token=False,
        )
        logger.info("Complete tokenizer loading...")
        model.config.use_cache = False
        calibration_datasets = datasets.load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1"
        )
        train_data = CustomJsonDataset(
            calibration_datasets["train"],
            tokenizer,
            block_size=min(training_args.model_max_length, 2048),
        )

        trainable_parameters = [model.R1.weight] + [
            model.model.layers[i].self_attn.R2.weight
            for i in range(model.config.num_hidden_layers)
        ]
        model.seqlen = training_args.model_max_length
        optimizer = SGDG(trainable_parameters, lr=training_args.learning_rate, stiefel=True)
        MyTrainer = Trainer
        # Use FSDP for 70B rotation training
        

        trainer = MyTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=None,
            data_collator=default_data_collator,
            optimizers=(optimizer, None),
        )
        torch.distributed.barrier()

        trainer.train()
        
        cpu_state = trainer.model.state_dict()

        R_dict = {
            key.replace(".weight", ""): value
            for key, value in cpu_state.items()
            if "R1.weight" in key or "self_attn.R2" in key
        }
        if local_rank == 0:
            os.makedirs(model_args.output_rotation_path, exist_ok=True)
            path = os.path.join(model_args.output_rotation_path, "R.bin")
            torch.save(
                R_dict,
                path,
            )
        dist.barrier()


if __name__ == '__main__':
    main()
