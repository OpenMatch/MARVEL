'''
From the ClueWeb22 data get the raw data for ClueWeb22-MM
'''

import argparse
import argparse
import csv
import json
import multiprocessing
import os

from bs4 import BeautifulSoup
from ClueWeb22Api import ClueWeb22Api
from tqdm import tqdm


def read_csv_to_dict(csv_file):
    data_dict = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 2:
                key = row[0]
                value = row[1]
                data_dict[key] = value

    return data_dict

def chunk_dict(input_dict, chunk_size):
    keys = list(input_dict.keys())
    num_chunks = len(keys) // chunk_size + (1 if len(keys) % chunk_size != 0 else 0)

    chunked_dicts = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk_keys = keys[start_idx:end_idx]
        chunk = {key: input_dict[key] for key in chunk_keys}
        chunked_dicts.append(chunk)

    return chunked_dicts

def generate_cw22id_list(cw22_filenum_dict):
    cw22id_list = []
    for k,v in cw22_filenum_dict.items():
        prefix = k
        count = int(v)
        for j in range(count):
            cw22id = f"clueweb22-{prefix}-{j:05d}"
            cw22id_list.append(cw22id)


    return cw22id_list

def chunk_list(lst, num_chunks):
    avg_chunk_size = len(lst) // num_chunks
    chunks = [lst[i:i + avg_chunk_size] for i in range(0, len(lst), avg_chunk_size)]
    return chunks

def process_chunk(chunk, root_path, results, process_id, total_processes):
    pbar = tqdm(chunk, position=process_id, desc=f"Process {process_id + 1}/{total_processes}")

    # Used to collect results in each process
    local_results = []

    for cw22id in pbar:
        cur_list = get_image_altText_and_surround(cw22id, root_path)
        local_results.extend(cur_list)

    # Adding local results to a shared results list
    results.extend(local_results)
    pbar.close()

def get_image_altText_and_surround(cw22id, root_path):
    clueweb_api = ClueWeb22Api(cw22id, root_path)
    all_node = clueweb_api.get_node_features()
    all_node_id_list = []
    for node in all_node:
        nid = int(node['node_id'])
        all_node_id_list.append(nid)

    # primary node
    node_with_text = clueweb_api.get_node_features_with_text()
    text_node_id_list = []
    text_node_dict = {}
    for node in node_with_text:
        nid = int(node['id'])
        text = node['text']
        text_node_id_list.append(nid)
        text_node_dict[nid] = text

    # image node
    image_text_dict = {}
    no_id_samples = []
    try:
        html_string = clueweb_api.get_html_from_warc()
        soup = BeautifulSoup(html_string, 'html.parser')
        img_tags = soup.find_all('img')
        for img in img_tags:
            alt = img.get('alt', None)
            src = img.get('src', None)
            id_attr = img.get('data-dcnode-id', None)
            if id_attr is not None:
                try:
                    id = int(id_attr)
                    unique_key = f"{cw22id}_{id}"  # concat cw22id and id
                    image_text_dict[unique_key] = {'src': src, 'alt': alt, 'nodeid': id, 'cw22id': cw22id}
                except ValueError:
                    print(f"Warning: Invalid data-dcnode-id attribute '{id_attr}' for image tag.")
                    continue
            else:
                print(f"Warning: Image tag without data-dcnode-id attribute in {cw22id}.")
                no_id_samples.append({'src': src, 'alt': alt, 'nodeid': None, 'cw22id': cw22id})
    except Exception as e:
        print(f"Error processing {cw22id}: {str(e)}")
    image_node_id_list = []
    for node in image_text_dict.values():
        nid = int(node['nodeid'])
        image_node_id_list.append(nid)

    # filter
    filter_node_id_list = [item for item in all_node_id_list if item in text_node_id_list or item in image_node_id_list]

    # Saves the surround text id for each image
    surround_id = {}
    for value in image_node_id_list:
        try:
            index = filter_node_id_list.index(value)
            left_values = []
            left_index = index - 1
            while left_index >= 0 and filter_node_id_list[left_index] not in image_node_id_list:
                left_values.insert(0, filter_node_id_list[left_index])
                left_index -= 1
            right_values = []
            right_index = index + 1
            while right_index < len(filter_node_id_list) and filter_node_id_list[right_index] not in image_node_id_list:
                right_values.append(filter_node_id_list[right_index])
                right_index += 1
            surround_id[value] = left_values + right_values
        except ValueError:
            print(f"Warning: Value {value} not found in filter_node_id_list for image {cw22id}.")
    # save the surrounding text
    for id,slist in surround_id.items():
        unique_key = f"{cw22id}_{id}"
        text_list = []
        for sid in slist:
            text = text_node_dict[sid]
            text_list.append(text)
        image_text_dict[unique_key]['surround'] = text_list

    # the final result
    img_text_list = []
    for value in image_text_dict.values():
        img_text_list.append(value)
    for value in no_id_samples:
        img_text_list.append(value)
    return img_text_list


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--file_num", type=int, default=0)
    parser.add_argument("--output_path", type=str, default='/clueweb-imgtext/raw')
    parser.add_argument('--csv_file_path',type=str,default='../ClueWeb22/record_counts/html/en00_counts.csv')
    parser.add_argument('--root_path',type=str,default='../ClueWeb22')
    args = parser.parse_args()


    # clueweb22 data
    cw22_filenum_dict = read_csv_to_dict(args.csv_file_path)
    cw22_filenum_dict_list = chunk_dict(cw22_filenum_dict, 100)
    cw22id_list = generate_cw22id_list(cw22_filenum_dict_list[args.file_num])

    print('--------load_files--------------')
    print(len(cw22id_list))
    print('--------load_files_finished--------------')

    root_path = args.root_path
    manager = multiprocessing.Manager()
    results = manager.list()

    num_processes = 70
    # num_processes = multiprocessing.cpu_count()   # save 1/4
    chunks = chunk_list(cw22id_list, num_processes)
    processes = []

    for process_id, chunk in enumerate(chunks):
        p = multiprocessing.Process(target=process_chunk, args=(chunk, root_path, results, process_id, num_processes))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print('--------get_data--------------')
    print(len(results))
    print('--------get_data_finished--------------')

    results = list(results)
    output_json_file = f'raw_img_text_{args.file_num}.json'
    output_file_path = os.path.join(args.output_path,output_json_file)
    with open(output_file_path, "w") as json_file:
        json.dump(results, json_file)

    print('--------all_finished--------------')





if __name__ == '__main__':
    main()