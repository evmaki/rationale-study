from os import listdir
from os.path import isfile, join
import json

datapath = './ConsolidatedAnnotation'
files = [f for f in listdir(datapath) if isfile(join(datapath, f))]
consolidated = []

for f in files:
    with open(f'{datapath}/{f}') as fp:
        f_json = json.load(fp)

        for item in f_json:
            if 'annotations' in item:
                for annotation in item['annotations']:
                    annotation_content = json.loads(annotation['annotationData']['content'])

                    new_annotation =  {
                        'workerId': annotation['workerId'],
                        'datasetObjectId': item['datasetObjectId'],
                        'text': item['dataObject']['content'],
                        'label': annotation_content['category']['label'],
                        'rationale': annotation_content['rationale']
                    }

                    consolidated.append(new_annotation)
            else:
                print(f'No annotations found in {datapath}/{f}')

with open('./consolidated_annotations.json', 'w') as f:
    json.dump(consolidated, f)