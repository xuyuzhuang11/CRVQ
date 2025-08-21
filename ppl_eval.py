import argparse
import torch
from main import perplexity_eval, get_model
from src.datautils import get_loaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--quant_path", type=str, default=None, help="Path to quantized model (if any)")
    parser.add_argument("--seqlen", type=int, default=4096, help="Model sequence length")

    args = parser.parse_args()

    device = torch.device(args.device)
    args.devices = [device]

    model = get_model(args.model_path, args.quant_path).eval()

    print("\n============ Evaluating perplexity... ============")
    torch.cuda.reset_peak_memory_stats()
    datasets = ["wikitext2", "c4"]


    eval_args = argparse.Namespace()
    eval_args.devices = [torch.device("cuda:0")]
    eval_args.offload_activations = True

    for dataset in args.datasets:
        testloader = get_loaders(
            dataset,
            seed=0,
            model_path=args.model_path,
            seqlen=args.seqlen,
            eval_mode=True,
            use_fast_tokenizer=False,
            trust_remote_code=False,
        )
        args.dataset_name = dataset
        perplexity_eval(model, testloader, eval_args)

    print(f"eval: {torch.cuda.max_memory_allocated()=:,}")

if __name__ == "__main__":
    main()
