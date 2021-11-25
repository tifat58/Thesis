import cv2


def scaleRadius(img, scale):
    x = img[int(img.shape[0] / 2), :, :].sum(1)
    #     print(x)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    #     print(r, s)
    try:
        ret_img = cv2.resize(img, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
    except:
        ret_img = None

    return ret_img, r, s


import matplotlib.pyplot as plt


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# returns kaggle
def get_kaggle_retinopathy_dataset_info(req_set='train'):
    kaggle_csv_file_path = '/mnt/sda/haal02-data/Kaggle-Retinopathy-Datasets/dr-data/trainLabels.csv'
    # truncated one
    #     kaggle_csv_file_path = '/mnt/sda/haal02-data/Kaggle-Retinopathy-Datasets/dr-data/truncated_trainLabels.csv'
    #     kaggle_image_path = '/mnt/sda/haal02-data/Kaggle-Retinopathy-Datasets/dr-data/train/'
    kaggle_image_path = '/mnt/sda/haal02-data/Kaggle-Retinopathy-Datasets/dr-data/train-data-processed1/'
    data_file = pd.read_csv(kaggle_csv_file_path, header=0)
    print(len(data_file))
    # print(data_file.head())
    db_len = 0
    start = 0
    end = 0
    # 0.7 train, 0.15 val, 0.15 test
    if req_set == 'train':
        start = 0
        end = len(data_file)
        # end = 24588
    #         end = int(0.7 * len(data_file))
    elif req_set == 'val':
        start = int(0.7 * len(data_file))
        # end = 29857
        #         end = int(0.85 * len(data_file))
        end = len(data_file) - 1
    elif req_set == 'test':
        start = int(0.85 * len(data_file))
        end = len(data_file) - 1

    elif req_set == 'full':
        start = 0
        end = len(data_file) - 1

    name_list = []
    label_list = []

    for i in range(start, end):
        file_name = data_file.iloc[i][0] + '.jpeg'

        if os.path.isfile(os.path.join(kaggle_image_path, file_name)):
            if file_name != '492_right.jpeg':
                name_list.append(os.path.join(kaggle_image_path, file_name))
                label_list.append(data_file.iloc[i][1])
        else:
            print()

    # for idx, row in data_file.iterrows():
    #     if os.path.isfile(os.path.join(fgadr_image_path, row[0])):
    #         name_list.append(os.path.join(fgadr_image_path, row[0]))
    #         label_list.append(row[1])
    #     else:
    #         print()

    return name_list, label_list


def get_retinopathy_dataset_info(req_set='train'):
    fgadr_root_path = '/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set'
    fgadr_csv_file_name = 'DR_Seg_Grading_Label.csv'
    fgadr_image_path = '/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set/Original_Images'

    data_file = pd.read_csv(os.path.join(fgadr_root_path, fgadr_csv_file_name), header=None)

    db_len = 0
    start = 0
    end = 0
    # 0.7 train, 0.15 val, 0.15 test
    if req_set == 'train':
        start = 0
        end = 1450
    elif req_set == 'val':
        start = 1450
        #         end = 1565
        end = len(data_file) - 1
    elif req_set == 'test':
        start = 1565
        end = len(data_file) - 1

    elif req_set == 'full':
        start = 0
        end = len(data_file) - 1

    name_list = []
    label_list = []

    for i in range(start, end):
        if os.path.isfile(os.path.join(fgadr_image_path, data_file.iloc[i][0])):
            name_list.append(os.path.join(fgadr_image_path, data_file.iloc[i][0]))
            label_list.append(data_file.iloc[i][1])
        else:
            print()

    # for idx, row in data_file.iterrows():
    #     if os.path.isfile(os.path.join(fgadr_image_path, row[0])):
    #         name_list.append(os.path.join(fgadr_image_path, row[0]))
    #         label_list.append(row[1])
    #     else:
    #         print()

    return name_list, label_list

