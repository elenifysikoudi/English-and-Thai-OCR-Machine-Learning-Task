import os
import pandas as pd
import argparse
import random

parser = argparse.ArgumentParser(description='Split data into train, validation and test files.')
parser.add_argument("train_language", type=str, help="The options are English, Thai, or both for what the training data will be.")
parser.add_argument("train_font", type=str, help="The options are normal, bold, bold_italic, italic, or all for what the training data will be.")
parser.add_argument("train_dpi", type=str, help="The options are 200, 300, 400 or all for what the training data will be.")
parser.add_argument("directory", default="none",help = "The path of the dataset /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet .")
parser.add_argument("--test_language", type=str, default="none", help="The options are English, Thai, or both for what the testing data will be.")
parser.add_argument("--test_font", type=str, default="none",help="The options are normal, bold, bold_italic, italic, or all for what the testing data will be.")
parser.add_argument("--test_dpi", type=str, default="none",help="The options are 200, 300, 400 or all for what the testing data will be.")
args = parser.parse_args()

def collect_from_folder(language, font, dpi, directory, images):
    lang_dir = os.path.join(directory,language)
    walking = os.walk(lang_dir)
    dpi_options = ["200", "300", "400"] if dpi == "all" else [str(dpi)]
    for root,_,files in walking:
        if font != "all":
            path_parts = root.split(os.sep)
            language_idx = path_parts.index(language)
            if any(d in root for d in dpi_options) and path_parts[-1] == font:
                ocr_number = path_parts[language_idx + 1 ]
                for file in files:
                    if file.endswith('.bmp'):
                        images.append((os.path.join(root,file), ocr_number))
        if font == "all" : 
           if any(d in root for d in dpi_options):
                path_parts = root.split(os.sep)
                language_idx = path_parts.index(language)
                ocr_number = path_parts[language_idx + 1 ]
                for file in files:
                    if file.endswith('.bmp'):
                        images.append((os.path.join(root,file), ocr_number))


def split_data(train_language, train_font, train_dpi, directory, test_language, test_font, test_dpi):
    train_images = []
    test_images = []
    valid_size = 0.1
    test_size = 0.1
    
    if train_language == "both":
        collect_from_folder('English',train_font, train_dpi, directory, train_images)
        collect_from_folder('Thai',train_font, train_dpi, directory, train_images)
    else:
        collect_from_folder(train_language,train_font, train_dpi, directory, train_images)

    random.shuffle(train_images)

    if test_language == "none" or test_font == "none" or test_dpi == "none":
        total_train_size = len(train_images)
        test_split = int(total_train_size * test_size)
        test_set = train_images[:test_split]
        valid_split = int(total_train_size*(test_size + valid_size))
        valid_set = train_images[test_split:valid_split]
        train_set = train_images[valid_split:]
        
        
    else:

        if test_language == "both":
            collect_from_folder('English', test_font, test_dpi, directory, test_images)
            collect_from_folder('Thai', test_font, test_dpi, directory, test_images)
        else:
            collect_from_folder(test_language, test_font, test_dpi, directory, test_images)

        random.shuffle(test_images)

        total_train_size = len(train_images)
        test_limit = int(total_train_size * test_size)
        test_set = test_images[:test_limit]
        
        total_train_size = len(train_images)
        valid_split = int(total_train_size * valid_size)

        valid_set = train_images[:valid_split]
        train_set = train_images[valid_split:]
    

    with open("train_file.txt",'w') as train:
        for path,ocr_number in train_set:
            train.write(f"{path},{ocr_number}\n")

    with open("test_file.txt",'w') as test:
        for path,ocr_number in test_set:
            test.write(f"{path},{ocr_number}\n")

    with open("valid_file.txt",'w') as valid:
        for path,ocr_number in valid_set:
            valid.write(f"{path},{ocr_number}\n")

split_data(args.train_language, args.train_font, args.train_dpi, args.directory, args.test_language, args.test_font , args.test_dpi )

    