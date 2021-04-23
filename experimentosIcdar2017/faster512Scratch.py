from icevision.all import *
import requests
import tarfile
from utils.utils import *

torch.cuda.set_device(2)
path='./../icdar2017All'

class_map = ClassMap(['formula', 'figure','table'])

parserTrain = parsers.voc(annotations_dir=Path(path+'/train/xmls'), images_dir=Path(path+'/train/JPEGImages'),class_map=class_map)
parserValid = parsers.voc(annotations_dir=Path(path+'/test/xmls'), images_dir=Path(path+'/test/JPEGImages'),class_map=class_map)

train_records, _ = parserTrain.parse(data_splitter=RandomSplitter([1, 0]))
_, valid_records = parserValid.parse(data_splitter=RandomSplitter([0, 1]))

presize = 600
size = 512

valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size), tfms.A.Normalize()])
train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=size, presize=presize,shift_scale_rotate=None,crop_fn=None, horizontal_flip=None, pad=None), tfms.A.Normalize()])

train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

model = faster_rcnn.model(num_classes=len(class_map),pretrained=False)

train_dl = faster_rcnn.train_dl(train_ds, batch_size=8, num_workers=0, shuffle=True)
valid_dl = faster_rcnn.valid_dl(valid_ds, batch_size=8, num_workers=0, shuffle=False)

metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

saveM=SaveModelMAP(monitor='COCOMetric',fname='model-ice-fasterIcdarScratch-v1')
lrReduce=ReduceLRMAP(monitor='COCOMetric',patience=3, factor=10.0, min_lr=0)
early=EarlyStoppingMAP(monitor='COCOMetric',patience=5)

learn = faster_rcnn.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics,cbs=[saveM,lrReduce,early])

learn.fine_tune(100,freeze_epochs=2)

learn.save('ice-fasterIcdarScratch-v1')
