from run_mia_unified import *
import os
import argparse

def list_of_strings(arg):
    return arg.split(',')

def load_json_data(folder):
    json_files = [pos_json for pos_json in os.listdir(folder) if pos_json.endswith('.json') and 'raw' not in pos_json and 'args' not in pos_json and 'roberta' not in pos_json and 'entropy' not in pos_json and 'logrank' not in pos_json]
    # print(json_files)
    outputs = []
    for j in json_files:
        with open(os.path.join(folder, j)) as json_file:
            outputs.append(json.load(json_file))
    
    return outputs

def plot_low_fpr_curves(experiments, SAVE_FOLDER, fpr_value=0.1):
    # first, clear plt
    plt.clf()

    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["metrics"]
        fpr_idx = [i for i, fpr in enumerate(metrics["fpr"]) if fpr <= fpr_value][-1]
        # fpr_idx = np.where(metrics["fpr"] <= fpr_value)[0][-1]
        plt.plot(metrics["fpr"][:fpr_idx+1], metrics["tpr"][:fpr_idx+1], label=f"{experiment['name']}", color=color)
        # , roc_auc={metrics['roc_auc']:.3f}
        # print roc_auc for this experiment
        print(f"{experiment['name']} roc_auc: {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, fpr_value])
    plt.ylim([0.0, 0.3])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves (GPT2) for d_f = imdb at FPR <= {fpr_value}')
    # plt.legend(loc="lower right", fontsize=6)
    plt.legend(fontsize=6)
    plt.savefig(f"{SAVE_FOLDER}/roc_curves_fpr_{fpr_value}.png")

def plot_asr_curves(experiments, orig, orig_ref, SAVE_FOLDER):
    # first, clear plt
    plt.clf()
    if type(experiments) == dict:
        for experiment, o, o_ref, color in zip(experiments, orig, orig_ref, COLORS):
            metrics = experiment["asr"]
            metric_orig = o["asr"]
            metric_orig_ref = o_ref["asr"]
            name = experiment['name'].split("_threshold")
            plt.bar(name[0], metrics["rate"], color=color)
            plt.bar(name[0], metric_orig["rate"])
            plt.bar(name[0], metric_orig_ref["rate"])
    else:
        asr = {}
        orig_asr ={}
        orig_ref_asr = {}
        for exp in experiments[0]:
            name = exp['name'].split("_threshold")
            asr[name[0]] = list()
            orig_asr[name[0]] = list()
            orig_ref_asr[name[0]] = list()
        
        for experiment in experiments:
            for exp in experiment:
                name = exp['name'].split("_threshold")
                asr[name[0]].append(exp["asr"]["rate"])
        for experiment in orig:
            for exp in experiment:
                name = exp['name'].split("_threshold")
                orig_asr[name[0]].append(exp["asr"]["rate"])
        for experiment in orig_ref:
            for exp in experiment:
                name = exp['name'].split("_threshold")
                orig_ref_asr[name[0]].append(exp["asr"]["rate"])

        mean_asr = [np.mean(sr) for sr in asr.values()]
        std_asr = [np.std(sr) for sr in asr.values()]
        mean_orig_asr = [np.mean(sr) for sr in orig_asr.values()]
        std_orig_asr = [np.std(sr) for sr in orig_asr.values()]
        mean_orig_ref_asr = [np.mean(sr) for sr in orig_ref_asr.values()]
        std_orig_ref_asr = [np.std(sr) for sr in orig_ref_asr.values()]
        X_axis = np.arange(len(asr.keys()))
        plt.bar(X_axis - 0.2, mean_asr, 0.2, yerr=std_asr, capsize=2, label='Unlearned')
        plt.bar(X_axis, mean_orig_ref_asr, 0.2, yerr=std_orig_ref_asr, capsize=2, label='Unlearned_ref_original')
        plt.bar(X_axis + 0.2, mean_orig_asr, 0.2, yerr=std_orig_asr, capsize=2, label='Original')

    plt.ylabel('Attack Success Rate')
    plt.xticks(X_axis, asr.keys(), rotation=45, fontsize=6)
    plt.title(f'ASR Curves for d_f = Harry Potter books')
    plt.legend()
    plt.savefig(f"{SAVE_FOLDER}/asr_curves.png")
    plt.savefig(f"{SAVE_FOLDER}/asr_curves.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, default="results")
    parser.add_argument("--unlearn_list", type=list_of_strings, default= None)
    parser.add_argument("--original_list", type= list_of_strings, default= None)
    parser.add_argument("--original_ref_list", type= list_of_strings, default= None)
    args = parser.parse_args()
    if args.unlearn_list == None or args.original_list == None:
        experiments = load_json_data(args.save_folder)
    else:
        experiments = []
        orig = []
        orig_ref = []
        for path in args.unlearn_list:
            experiments.append(load_json_data(path))
        for path in args.original_list:
            orig.append(load_json_data(path))
        for path in args.original_ref_list:
            orig_ref.append(load_json_data(path))
    # plot_low_fpr_curves(experiments, fpr_value=0.1)
    plot_asr_curves(experiments, orig, orig_ref, args.save_folder)
