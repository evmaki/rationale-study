import json
import os
import sys

# TODO define these classes
classes = ["null", "word choice"]

def class_input(class_key):
    print(f'{class_key}: ', end='') 
    label = input()

    if label not in classes:
        print(f'{label} is not a valid class.')
        label = class_input(class_key)

    return label

try:
    with open(f'./consolidated_annotations.json') as fp:
        annotations = json.load(fp)
        encoded_annotations = []

        for item in annotations:
            if 'class-1' not in item and 'class-2' not in item and 'confidence' not in item:
                print('-'*100)
                print(f'text: {item["text"]}\n')
                print(f'rationale: {item["rationale"]}\n')
                class1 = class_input('class-1')
                class2 = class_input('class-2')

                item['class-1'] = class1 if class1 != 'null' else None
                item['class-2'] = class2 if class2 != 'null' else None
            
            encoded_annotations.append(item)
        
        with open('./coded_consolidated_annotations.json', 'w') as f:
            json.dump(encoded_annotations, f)

except KeyboardInterrupt:
    print('Interrupted')

    with open('./coded_consolidated_annotations.json', 'w') as f:
        json.dump(encoded_annotations, f)

    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)