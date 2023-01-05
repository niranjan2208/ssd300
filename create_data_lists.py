from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(voc07_path='../COCO_Dataset/COCO_2017_PascalVOC_Dataset_Exp',
                      voc12_path='../COCO_Dataset/COCO_2017_PascalVOC_Dataset_Exp',
                      output_folder='../input_json_exp')
