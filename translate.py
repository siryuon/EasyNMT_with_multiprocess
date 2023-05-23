import json
import torch
from easynmt import EasyNMT

ckpt = YOUR_OWN_MODEL_CKPT
data_path = YOUR_OWN_DATA_PATH
output_path = YOUR_OWN_OUTPUT_PATH
json_data = []
translated_data = []
resume = False

with open(data_path, 'r') as fp:
    for line in fp:
        json_data.append(json.loads(line))

def initialize_model():
    global model
    torch.cuda.init()
    model = EasyNMT(ckpt, max_loaded_model=10)

def translate_data(data):
    '''
    Fill your own translation code
    
    For example:
    instruction = data['instruction']
    response = data['response']
    context = data.get('context')

    trans_instruction = model.translate(instruction, source_lang='en', target_lang='ko')
    trans_response = model.translate(response, source_lang='en', target_lang='ko')
    
    try:
        trans_context = model.translate(context, source_lang='en', target_lang='ko')
    except:
        trans_context = None
    if trans_context:
        translated = {
            'instruction': trans_instruction,
            'context': trans_context,
            'response': trans_response
        }
    else:
        translated = {
            'instruction': trans_instruction,
            'response': trans_response
        }
    
    return translated
    '''
    return None

def process_data(idx):
    data = json_data[idx]
    torch.cuda.empty_cache()
    translated = translate_data(data)
    translated_data.append(translated)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    initialize_model()

    with torch.multiprocessing.Pool(initializer=initialize_model, processes=4) as pool:
        pool.map(process_data, range(len(json_data)))

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, indent='\t', ensure_ascii=False)
