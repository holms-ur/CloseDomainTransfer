from icevision.all import *
import requests
import tarfile
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils.utils import *

torch.cuda.set_device(1)
path='./../FUNSD'

class_map = ClassMap(['header','question', 'answer','other'])

parserTrain = parsers.coco(annotations_file=Path(path+'/train.json'), img_dir=Path(path+'/training_data/images'),mask=False)
parserValid = parsers.coco(annotations_file=Path(path+'/test.json'), img_dir=Path(path+'/testing_data/images'), mask=False)

train_records, _ = parserTrain.parse(data_splitter=RandomSplitter([1, 0]))
_, valid_records = parserValid.parse(data_splitter=RandomSplitter([0, 1]))

presize = 600
size = 512

valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size), tfms.A.Normalize()])
train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=size, presize=presize,shift_scale_rotate=None,crop_fn=None, horizontal_flip=None, pad=None), tfms.A.Normalize()])

train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

model = faster_rcnn.model(num_classes=6)
fastai.load_model('../models/ice-faster-512-v2.pth',model,None,with_opt=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(class_map))


train_dl = faster_rcnn.train_dl(train_ds, batch_size=8, num_workers=0, shuffle=True)
valid_dl = faster_rcnn.valid_dl(valid_ds, batch_size=8, num_workers=0, shuffle=False)

metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

saveM=SaveModelMAP(monitor='COCOMetric',fname='model-ice-fasterFUNSDFineAll-v1')
lrReduce=ReduceLRMAP(monitor='COCOMetric',patience=3, factor=10.0, min_lr=0)
early=EarlyStoppingMAP(monitor='COCOMetric',patience=5)

learn = faster_rcnn.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics,cbs=[saveM,lrReduce,early])

learn.fine_tune(15,freeze_epochs=2)

learn.save('ice-fasterFUNSDFineAll-v1')
