import os
# import argparse

ROOT_DIR = os.environ["HOME"]
# ROOT_DIR_LOCAL = '/state/share1'  # for share storage

ROOT_DIR_LOCAL = '/projets/AS08960'  # for local storage in taylor-1, taylor-3, taylor-4, taylor-5


ROOT_DIR_LOC_PRJ = '/projets/AS08960'  # for local storage in taylor-1, taylor-3, taylor-4, taylor-5


CURRENT_DIR = os.path.abspath(os.getcwd())

CURRENT_LOCAL_DIR = ROOT_DIR_LOCAL + '/Biovid'
CURRENT_HOME_DIR = ROOT_DIR + '/Biovid'

DATASET_FOLDER = "/datasets"

LOCAL_SERVER = 1
LIVIA_SERVER = 2

# _FER_DATASET_PATH = ROOT_DIR + '/Downloads/PhD/FER/datasets'    # Path for local server
                # Path for livia server

WEIGHTS_FOLDER = "WeightFiles"

_BIOVID_DATASET_LOCAL_PATH = ROOT_DIR_LOC_PRJ + DATASET_FOLDER      # Path for Biovid dataset local server

# 1.original
SHARED_DIR = '/datasets/abaw8-workshop'  # for local storage in taylor-1, taylor-3, taylor-4, taylor-5
# 1. My local 
_DATASET_LOCAL_PATH = ROOT_DIR_LOCAL + DATASET_FOLDER
# 1. My home
_FER_DATASET_PATH = ROOT_DIR + DATASET_FOLDER

DATASET_PATH = _FER_DATASET_PATH

# PRETRAINED_WEIGHTS = CURRENT_HOME_DIR + '/pretrained-weights'

# ----------------- Source Dataset
SOURCE_DATASET_PATH = DATASET_PATH + '/RAF-DB_AffectNet_AFF-WILD2'

SOURCE_LABEL_PATH_TRAIN = DATASET_PATH + '/RAF-DB_AffectNet_AFF-WILD2/RAF-DB_AffectNet_AFF-WILD2/train.txt'

SOURCE_SS_C_EXPR_PATH_TRAIN = DATASET_PATH + '/RAF-DB_AffectNet_AFF-WILD2/RAF-DB_AffectNet_AFF-WILD2/src_select_samples_5k_bal.txt'
SOURCE_SS_RAF_PATH_TRAIN = DATASET_PATH + '/RAF-DB_AffectNet_AFF-WILD2/RAF-DB_AffectNet_AFF-WILD2/src_select_samples_1k_bal.txt'


SOURCE_LABEL_PATH_VAL = DATASET_PATH + '/RAF-DB_AffectNet_AFF-WILD2/RAF-DB_AffectNet_AFF-WILD2/val.txt'
SOURCE_LABEL_PATH_TEST = DATASET_PATH + '/RAF-DB_AffectNet_AFF-WILD2/RAF-DB_AffectNet_AFF-WILD2/test.txt'

SOURCE_CLASSES = DATASET_PATH + '/RAF-DB_AffectNet_AFF-WILD2/RAF-DB_AffectNet_AFF-WILD2/class_id.yaml'

#------------ Biovid Datasets
BIOVID = 'Biovid'
BIOVID_PATH = _BIOVID_DATASET_LOCAL_PATH + '/Biovid'

BIOVID_SOURCE_DATASET_PATH = DATASET_PATH + '/Biovid/sub_classes_tpt'

BIOVID_ALL_CAT_SUBS_PATH = _FER_DATASET_PATH + '/Biovid/sub_red_classes_img'
BIOVID_TAR_SUB_PATH = _FER_DATASET_PATH + '/Biovid/sub_red_classes_img/081609_w_40/'

BIOVID_VIDEO_LABEL_PATH = _FER_DATASET_PATH + '/Biovid/labels.txt'


BIOVID_SUBS_PATH = _FER_DATASET_PATH + '/Biovid/sub_red_classes_img'
BIOVID_RED_SUBS_FOLDER = 'sub_red_classes_img'


#------------ Stresst Dataset

STRESS = 'StressID'
STRESS_PATH = _FER_DATASET_PATH + '/StressID'
STRESS_ALL_SUBS_PATH = _FER_DATASET_PATH + '/StressID/sub_images'

STRESS_VIDEO_PATH = STRESS_PATH + '/Videos/2ea4/2ea4_Baseline.mp4'

# -- all labels
STRESS_ALL_LABEL_PATH = _FER_DATASET_PATH + '/StressID/all_sub_labels.txt'

STRESS_BIO_DATASET  = 'sub_physio_split'
STRESS_BIO_PATH =  _FER_DATASET_PATH + '/StressID/' + STRESS_BIO_DATASET

STRESS_AUDIO_DATASET  = 'sub_audio_split'
STRESS_AUDIO_PATH =  _FER_DATASET_PATH + '/StressID/' + STRESS_AUDIO_DATASET

# -- this label path is by assigning every subject a class

STRESS_ID_LABEL_VISUAL_PATH = _FER_DATASET_PATH + '/StressID/sub_N_source_classes.txt'

STRESS_SUBID_TO_SUBNAME_MAPPING = _FER_DATASET_PATH + '/StressID/sub_mapping_N_classes.txt'

STRESS_ID_LABEL_PHYSIO_PATH = _FER_DATASET_PATH + '/StressID/sub_N_source_class_physio.txt'

# -- all classes categorical for CAS class-aware
STRESS_ALL_CAT_SUBS_PATH = _FER_DATASET_PATH + '/StressID/sub_classes'

STRESS_ALL_CLASSES = ['N', 'S']

#------------ Target Datasets
# ----------------- Compund Expression ** Labeled ** ----------
C_EXPR_DB = 'C-EXPR-DB'
C_EXPR_LABELED_DATASET_PATH = DATASET_PATH + '/C-EXPR-DB'

C_EXPR_LABEL_PATH_TRAIN_FOLD_0 = DATASET_PATH + '/C-EXPR-DB/C-EXPR-DB/split-0/train.txt'
C_EXPR_LABEL_PATH_TRAIN_FOLD_1 = DATASET_PATH + '/C-EXPR-DB/C-EXPR-DB/split-1/train.txt'
C_EXPR_LABEL_PATH_TRAIN_FOLD_2 = DATASET_PATH + '/C-EXPR-DB/C-EXPR-DB/split-2/train.txt'
C_EXPR_LABEL_PATH_TRAIN_FOLD_3 = DATASET_PATH + '/C-EXPR-DB/C-EXPR-DB/split-3/train.txt'
C_EXPR_LABEL_PATH_TRAIN_FOLD_4 = DATASET_PATH + '/C-EXPR-DB/C-EXPR-DB/split-4/train.txt'

C_EXPR_LABEL_PATH_TEST_FOLD_0 = DATASET_PATH + '/C-EXPR-DB/C-EXPR-DB/split-0/test.txt'
C_EXPR_LABEL_PATH_TEST_FOLD_1 = DATASET_PATH + '/C-EXPR-DB/C-EXPR-DB/split-1/test.txt'
C_EXPR_LABEL_PATH_TEST_FOLD_2 = DATASET_PATH + '/C-EXPR-DB/C-EXPR-DB/split-2/test.txt'
C_EXPR_LABEL_PATH_TEST_FOLD_3 = DATASET_PATH + '/C-EXPR-DB/C-EXPR-DB/split-3/test.txt'
C_EXPR_LABEL_PATH_TEST_FOLD_4 = DATASET_PATH + '/C-EXPR-DB/C-EXPR-DB/split-4/test.txt'

C_EXPR_BE_CLASSES = DATASET_PATH + '/C-EXPR-DB/pseudo-labels/apvit/model_apvit/class_id.yml'
C_EXPR_COMP_CLASSES = DATASET_PATH + '/C-EXPR-DB/C-EXPR-DB/split-0/class_id.yaml'

PRETRAINED_C_EXPR_WEIGHT_PATH = DATASET_PATH + '/C-EXPR-DB/pseudo-labels/apvit/model_apvit'
PRETRAINED_C_EXPR_WEIGHT_FILE = 'model_apvit/logits_apvit_C-EXPR-DB'

# ----------------- Compund Expression ** Challenge ** ----------
C_EXPR_DB_CHALLENGE = 'C-EXPR-DB-CHALLENGE'
C_EXPR_DATASET_PATH = DATASET_PATH + '/C-EXPR-DB-CHALLENGE'
C_EXPR_CHALL_LABEL_PATH_FOLD_0 = DATASET_PATH + '/C-EXPR-DB-CHALLENGE/C-EXPR-DB-CHALLENGE/split-0/test.txt'

# ----------------- RAF-DB ----------
RAF_DATASET_CLS_PATH = DATASET_PATH + '/RAF/basic/Image'

RAF_DB = 'RAF'
RAF_DATASET_PATH = DATASET_PATH + '/RAF-DB-COMPOUND'
RAF_LABEL_PATH_TRAIN = DATASET_PATH + '/RAF-DB-COMPOUND/RAF-DB-COMPOUND/train.txt'
RAF_LABEL_PATH_VAL = DATASET_PATH + '/RAF-DB-COMPOUND/RAF-DB-COMPOUND/val.txt'
RAF_LABEL_PATH_TEST = DATASET_PATH + '/RAF-DB-COMPOUND/RAF-DB-COMPOUND/test.txt'

RAF_BE_CLASSES = DATASET_PATH + '/RAF-DB-COMPOUND/pseudo-labels/apvit/model_apvit/class_id.yml'
RAF_COMP_CLASSES = DATASET_PATH + '/RAF-DB-COMPOUND/RAF-DB-COMPOUND/class_id.yaml'
PRETRAINED_RAF_WEIGHT_PATH = DATASET_PATH + '/RAF-DB-COMPOUND/pseudo-labels/apvit/model_apvit'
PRETRAINED_RAF_WEIGHT_FILE = 'model_apvit/logits_apvit_RAF-DB-COMPOUND'

PRETRAINED_RAF_EMBEDDING_FILE = 'source_feature_class_representative_apvit.pkl'
PRETRAINED_RAF_LOGITS_FILE = 'source_per_class_average_logits_apvit.pkl'

# ----------------- BAH ----------
BAH = 'BAH_DB'

PRETRAINED_BAH_CONFIG_FOLDER = CURRENT_HOME_DIR +'/config_files'
BAH_DATASET_PATH = _BIOVID_DATASET_LOCAL_PATH + '/BAH_DB'

BAH_VIDEO_PATH = BAH_DATASET_PATH + '/wav/Videos/82553/Visite_1/82553_Question_1_2024-08-22_12-11-55_Video.mp4/82553_Question_1_2024-08-22_12-11-55_Video.wav'

BAH_TARGET_SPLIT_FOLDER = CURRENT_HOME_DIR + '/WeightFiles/target_split'

BAH_DATASET_FRAMES_PATH = BAH_DATASET_PATH + '/cropped-aligned-faces'

BAH_PATH_TRAIN = _BIOVID_DATASET_LOCAL_PATH + '/BAH_DB/split-frames/train.txt'
BAH_PATH_TRAIN_RB = DATASET_PATH + '/BAH_DB/split-frames/train_rb_20k.txt'

BAH_PATH_VAL = DATASET_PATH + '/BAH_DB/split-frames/val.txt'
BAH_PATH_TEST = DATASET_PATH + '/BAH_DB/split-frames/test.txt'

# ----------------- FERV39 ----------
FERV39k = 'FERV39k'

# PRETRAINED_BAH_CONFIG_FOLDER = CURRENT_HOME_DIR +'/config_files'
FERV39k_DATASET_PATH = _BIOVID_DATASET_LOCAL_PATH + '/FERV39k'

# FERV39k_TARGET_SPLIT_FOLDER = CURRENT_HOME_DIR + '/WeightFiles/target_split'

FERV39k_DATASET_FRAMES_PATH = FERV39k_DATASET_PATH + '/2_ClipsforFaceCrop'

FERV39k_PATH_TRAIN = FERV39k_DATASET_PATH + '/ferv39_train.txt'
FERV39k_PATH_TEST = FERV39k_DATASET_PATH + '/ferv39k_test.txt'

# ----------------- DFEW ----------
DFEW = 'DFEW'

# PRETRAINED_BAH_CONFIG_FOLDER = CURRENT_HOME_DIR +'/config_files'
DFEW_DATASET_PATH = _BIOVID_DATASET_LOCAL_PATH + '/DFEW'

# FERV39k_TARGET_SPLIT_FOLDER = CURRENT_HOME_DIR + '/WeightFiles/target_split'

DFEW_DATASET_FRAMES_PATH = DFEW_DATASET_PATH + '/DFEW-part1'

DFEW_PATH_TRAIN = DFEW_DATASET_PATH + '/dfew/dfew_train_set_1.txt'
DFEW_PATH_TEST = DFEW_DATASET_PATH + '/dfew/dfew_test_set_5.txt'


# ----------------- MAFW ----------
MAFW = 'MAFW'

# PRETRAINED_BAH_CONFIG_FOLDER = CURRENT_HOME_DIR +'/config_files'
MAFW_DATASET_PATH = _BIOVID_DATASET_LOCAL_PATH + '/MAFW'

# FERV39k_TARGET_SPLIT_FOLDER = CURRENT_HOME_DIR + '/WeightFiles/target_split'

MAFW_DATASET_FRAMES_PATH = MAFW_DATASET_PATH

MAFW_PATH_TRAIN = MAFW_DATASET_PATH + '/Train_Test_Split_Frames_7cls/set_5/train.txt'
MAFW_PATH_TEST = MAFW_DATASET_PATH + '/Train_Test_Split_Frames_7cls/set_5/test.txt'

# ------------------- Pretrained Paths -----------
BAH_SOURCE_PRETRAINED_PATH = '/datasets/neurips25/shared_weights/BAH_DB'

BAH_SOURCE_VIT_PRETRAINED_MODEL = BAH_SOURCE_PRETRAINED_PATH + '/apvit/id_6283723100_BAH_model_apvit__epochs_60_lr_0.003_p_train_1.-model_apvit-ds_BAH_DB/model_apvit_xx_best_for_xx_BAH_DB__6283723100_BAH_model_apvit__epochs_60_lr_0.003_p_train_1._xx_use-c-data-aug_False'
BAH_SOURCE_RESNET18_PRETRAINED_MODEL = BAH_SOURCE_PRETRAINED_PATH + '/resnet18/id_6283723290_model_resnet18__epochs_1_lr_0.009_p_train_0.1-model_resnet18-ds_BAH_DB/model_resnet18_xx_best_for_xx_BAH_DB__6283723290_model_resnet18__epochs_1_lr_0.009_p_train_0.1_xx_use-c-data-aug_False'

# ----------------- Target PL Pretrined Models ----------
ENET_VGAF = "enet_b0_8_best_vgaf"
ENET_AFEW = "enet_b0_8_best_afew"
ENET_B2_8 = "enet_b2_8"
ENET_B2_7 = "enet_b2_7"
APVIT = "apvit"

# ----------------- AUG Target DATASET for Model Selection - C-Expr ----------
AUG_C_VALID_EXPR_DB = 'AUG_COMP_VALID_C_EXPR_DB_cutmix_h'
AUG_C_VALID_EXPR_DATASET_PATH = DATASET_PATH + '/AUG_COMP_VALID_C_EXPR_DB_cutmix_h'

AUG_COMP_VALID_C_EXPR_DB_cutmix_h = AUG_C_VALID_EXPR_DATASET_PATH + '/AUG_COMP_VALID_C_EXPR_DB_cutmix_h/val.txt' 
AUG_COMP_VALID_C_EXPR_DB_cutmix_h_v = DATASET_PATH + '/AUG_COMP_VALID_C_EXPR_DB_cutmix_h_v/AUG_COMP_VALID_C_EXPR_DB_cutmix_h_v/val.txt'
AUG_COMP_VALID_C_EXPR_DB_cutmix_v = DATASET_PATH + '/AUG_COMP_VALID_C_EXPR_DB_cutmix_v/AUG_COMP_VALID_C_EXPR_DB_cutmix_v/val.txt'
AUG_COMP_VALID_C_EXPR_DB_mixup = DATASET_PATH + '/AUG_COMP_VALID_C_EXPR_DB_mixup/AUG_COMP_VALID_C_EXPR_DB_mixup/val.txt' 

# ----------------- AUG Target DATASET for Model Selection - RAF-DB ----------
AUG_C_VALID_RAFDB_COMPUND = 'AUG_COMP_VALID_RAFDB_COMPOUND_cutmix_h'
AUG_C_VALID_RAFDB_DATASET_PATH = DATASET_PATH + '/AUG_COMP_VALID_RAFDB_COMPOUND_cutmix_h'

AUG_COMP_VALID_RAFDB_DB_cutmix_h = AUG_C_VALID_RAFDB_DATASET_PATH + '/AUG_COMP_VALID_RAFDB_COMPOUND_cutmix_h/val.txt' 

# ----------------- Distance Measures ----------

MMD_SIMILARITY = 1
COSINE_SIMILARITY = 2
KL_DIVERGENCE = 3
GRASSMANNIAN_DIST = 4


# ----------------- Fusion Type ----------
FUS_CONCAT = 1
FUS_CROSSATEN = 2
FUS_GATED = 3
FUS_MOE = 4

# ------------------------------------------

WEIGHTS_FOLDER = "WeightFiles"
BIOVID_N_SRC_WEIGHT_FILE = 'WeightFiles/lab_srcs78_cl77_082208w45_081714m36_112610w60_101908m61_071709w23_082014w24_110810m62_080209w26_101916m40_110614m42_____only'
ALL_SOURCES_FOLDER = "AllSources"

BIOVID_TRAIN_IMG_SRC_WEIGHTS = CURRENT_DIR + '/WeightFiles/lab_srcs78_082208w45_081714m36_112610w60_101908m61_071709w23_082014w24_110810m62_080209w26_101916m40_110614m42_____only_load.pt'
BIOVID_TRAIN_BIO_SRC_WEIGHTS = CURRENT_DIR + "/WeightFiles/bio_lab_srcs78_w45_m36_w60_m61_w23_w24_m62_w26_m40_m42_____only.pth"
MODEL_FUS_PATH = CURRENT_DIR + "/lab_srcs78_w45_m36_w60_m61_w23_w24_m62_w26_m40_m42_____only_fus_load.pt"

COMET_API_KEY = "eow2bmNwSPBKrx657Qfx43lW7"
COMET_WORKSPACE = "osamazeeshan"
COMET_LOG_CODE = True
COMET_DISABLED = False
COMET_PROJECT_NAME = "mm-vlm-tt"

# -------------------
