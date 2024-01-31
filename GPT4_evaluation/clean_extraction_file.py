import argparse, os, json
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--result_dir", type=str)
    args = parser.parse_args()
    debug = False
    n_jobs = 200
    output_dir = args.result_dir
    model_name, data_name = args.model_name, args.data_name

    # in data name I have checkpoint num
    this_model_data_save_dir = os.path.join(output_dir, model_name, data_name)
    for one_ckpt_folder in os.listdir(this_model_data_save_dir):
        ckpt_result_folder = os.path.join(this_model_data_save_dir, one_ckpt_folder)
        all_json_files = glob.glob(os.path.join(ckpt_result_folder, "*.json"))

        for one_json_file in all_json_files:
            with open(one_json_file, "r") as f:
                json_content = f.readlines()
            base_json_file_name = os.path.basename(one_json_file)
            extraction_output_file = os.path.join(
                ckpt_result_folder,
                base_json_file_name.replace(".json", "") + "_gpt4_extraction.jsonl",
            )
            one_cmd = "rm {}".format(extraction_output_file)
            os.system(one_cmd)
