# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PASCAL VOC datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import xml.etree.ElementTree

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds


_VOC_CITATION = """\
@misc{{pascal-voc-{year},
	author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
	title = "The {{PASCAL}} {{V}}isual {{O}}bject {{C}}lasses {{C}}hallenge {year} {{(VOC{year})}} {{R}}esults",
	howpublished = "http://www.pascal-network.org/challenges/VOC/voc{year}/workshop/index.html"}}
"""
_SBD_CITATION = """\
@InProceedings{{BharathICCV2011,
    author = "Bharath Hariharan and Pablo Arbelaez and Lubomir Bourdev and Subhransu Maji and Jitendra Malik",
    title = "Semantic Contours from Inverse Detectors",
    booktitle = "International Conference on Computer Vision (ICCV)",
    year = "2011"}}
"""

_VOC_DETECTION_DESCRIPTION = """\
This dataset contains the data from the PASCAL Visual Object Classes Challenge
{year}, a.k.a. VOC{year}, corresponding to the Classification and Detection
competitions.
A total of {num_images} images are included in this dataset, where each image
contains a set of objects, out of 20 different classes, making a total of
{num_objects} annotated objects.
In the Classification competition, the goal is to predict the set of labels
contained in the image, while in the Detection competition the goal is to
predict the bounding box and label of each individual object.
WARNING: As per the official dataset, the test set of VOC2012 does not contain
annotations.
"""

_VOC_SEGMENTATION_DESCRIPTION_COMMON = """\
This dataset contains the data from the PASCAL Visual Object Classes Challenge
{year}, a.k.a. VOC{year}, corresponding to the Segmentation competition.
A total of {num_images} images are included in this dataset, where each image
has corresponding class and instance masks which label each pixel with a class
between 1 and 20, or an instance id. Label 0 is reserved and is used to mark the
background of images. Likewise, label 255 is reserved and is used to label
regions of the image which should be ignored (the boundaries between
classes/instances).
"""
_VOC_SEGMENTATION_DESCRIPTION_2007 = _VOC_SEGMENTATION_DESCRIPTION_COMMON.format(
    year=2007, num_images=632)
_VOC_SEGMENTATION_DESCRIPTION_2012 = _VOC_SEGMENTATION_DESCRIPTION_COMMON.format(
    year=2012, num_images=12031) + \
"""\
The splits "train" amd "validation" are the original training and validation
splits from VOC2012. The VOC2012 test split is not included as it does not have
publicly available annotations.
Also included are splits "sbd_train" and "sbd_validation", containing data from
the Semantic Boundaries Dataset. SBD is a superset of VOC2012; "sbd_train" and
"sbd_validation" contain all the images of the training and validation splits of
SBD, but with images already included in VOC2012 "train" and "validation"
removed, so that the four splits "train", "validation", "sbd_train", and
"sbd_validation" do not overlap. The commonly used "train_aug" split can be
obtained by combining the "train", "sbd_train", and "sbd_validation" splits.
"""

_VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc{year}/"
# Original site, it is down very often.
# _VOC_DATA_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc{year}/"
# Data mirror:
_VOC_DATA_URL = "http://pjreddie.com/media/files/"
_SBD_DATA_URL = "https://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/"

_VOC_LABELS = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
_VOC_POSES = (
    "frontal",
    "rear",
    "left",
    "right",
    "unspecified",
)


class IDGenerator(object):
  """A helper class to generate example IDs."""

  def __init__(self, filename, image_set_relative_path):
    self.filename = filename
    self.image_set_relative_path = image_set_relative_path

  def generate(self, paths):
    image_set_path = os.path.join(paths[self.filename], self.image_set_relative_path)
    with tf.io.gfile.GFile(image_set_path, "r") as f:
      for line in f:
        yield line.strip()

class IncludeExcludeIDGenerator(object):
  """
  An ID generator which yields every ID from `include_generators` which would
  not be yielded by any generator of `exclude_generators`. This is necessary
  for the splits 'sbd_train' and 'sbd_validation' of 'VOC/2012-segmentation'.
  """

  def __init__(self, include_generators, exclude_generators):
    self.include_generators = include_generators
    self.exclude_generators = exclude_generators

  def generate(self, paths):
    exclude_ids = set(
        example_id
        for generator in self.exclude_generators
        for example_id in generator.generate(paths)
    )
    for generator in self.include_generators:
      for example_id in generator.generate(paths):
        if example_id not in exclude_ids:
          yield example_id


def _get_detection_objects(annotation_filepath):
  """Function to get all the detection objects from an annotation XML file."""
  with tf.io.gfile.GFile(annotation_filepath, "r") as f:
    root = xml.etree.ElementTree.parse(f).getroot()

    # Disable pytype to avoid attribute-error due to find returning
    # Optional[Element]
    # pytype: disable=attribute-error
    size = root.find("size")
    width = float(size.find("width").text)
    height = float(size.find("height").text)

    for obj in root.findall("object"):
      # Get object's label name.
      label = obj.find("name").text.lower()
      # Get objects' pose name.
      pose = obj.find("pose").text.lower()
      is_truncated = (obj.find("truncated").text == "1")
      is_difficult = (obj.find("difficult").text == "1")
      bndbox = obj.find("bndbox")
      xmax = float(bndbox.find("xmax").text)
      xmin = float(bndbox.find("xmin").text)
      ymax = float(bndbox.find("ymax").text)
      ymin = float(bndbox.find("ymin").text)
      yield {
          "label": label,
          "pose": pose,
          "bbox": tfds.features.BBox(
              ymin / height, xmin / width, ymax / height, xmax / width),
          "is_truncated": is_truncated,
          "is_difficult": is_difficult,
      }
    # pytype: enable=attribute-error


class VocPath(object):
  def __init__(self, filename, relative_path):
    self.filename = filename
    self.relative_path = relative_path

  def resolve(self, paths, example_id):
    return os.path.join(
        paths[self.filename], os.path.normpath(self.relative_path)
    ).format(example_id)


class VOCExampleBuilder(object):
  """A base class to build VOC examples."""

  def __init__(self, year=None, filename=None, image_path=None):
    if image_path is None:
      image_path = VocPath(filename,
          "VOCdevkit/VOC{}/JPEGImages/{{}}.jpg".format(year))
    self.image_path = image_path

  def build(self, paths, example_id):
    return {
        "image": self.image_path.resolve(paths, example_id),
        "image/filename": example_id + ".jpg",
    }


class DetectionExampleBuilder(VOCExampleBuilder):
  """A helper class to build VOC object detection examples."""

  def __init__(self, year=None, filename=None, image_path=None, annotation_path=None):
    super(DetectionExampleBuilder, self).__init__(year, filename, image_path)
    if annotation_path is None:
      annotation_path = VocPath(filename,
          "VOCdevkit/VOC{}/Annotations/{{}}.xml".format(year))
    self.annotation_path = annotation_path

  def build(self, paths, example_id):
    if self.annotation_path:
      annotation_filepath = self.annotation_path.resolve(paths, example_id)
      objects = list(_get_detection_objects(annotation_filepath))
      # Use set() to remove duplicates.
      labels = sorted(set(obj["label"] for obj in objects))
      labels_no_difficult = sorted(set(
          obj["label"] for obj in objects if obj["is_difficult"] == 0
      ))
    else:  # If the annotation path is False there are no annotations.
      objects = []
      labels = []
      labels_no_difficult = []
    return {
        **super(DetectionExampleBuilder, self).build(paths, example_id),
        "objects": objects,
        "labels": labels,
        "labels_no_difficult": labels_no_difficult,
    }

class SegmentationExampleBuilder(DetectionExampleBuilder):
  """A helper class to build VOC segmentation examples."""

  def __init__(self, year=None, filename=None, image_path=None, annotation_path=None,
      class_mask_path=None, instance_mask_path=None):
    super(SegmentationExampleBuilder, self).__init__(year, filename, image_path, annotation_path)
    if class_mask_path is None:
      class_mask_path = VocPath(filename,
          "VOCdevkit/VOC{}/SegmentationClass/{{}}.png".format(year))
    if instance_mask_path is None:
      instance_mask_path = VocPath(filename,
          "VOCdevkit/VOC{}/SegmentationObject/{{}}.png".format(year))
    self.class_mask_path = class_mask_path
    self.instance_mask_path = instance_mask_path

  @staticmethod
  def _load_png_segmentation_mask(path):
    """Load a '.png' segmentation mask, ignoring any colour map."""

    with tf.io.gfile.GFile(path, "rb") as fp:
      image = tfds.core.lazy_imports.PIL_Image.open(fp)
    return np.expand_dims(np.array(image, dtype=np.uint8), axis=-1)

  @staticmethod
  def _load_mat_segmentation_mask(path, key):
    """Load a '.mat' segmentation mask of the kind used in the SBD dataset."""

    record = tfds.core.lazy_imports.scipy.io.loadmat(path, struct_as_record=True)
    return np.expand_dims(record[key]["Segmentation"][0][0], axis=-1)

  def build(self, paths, example_id):
    class_mask_filepath = self.class_mask_path.resolve(paths, example_id)
    instance_mask_filepath = self.instance_mask_path.resolve(paths, example_id)
    class_mask = (
        self._load_mat_segmentation_mask(class_mask_filepath, key="GTcls")
        if class_mask_filepath.endswith(".mat")
        else self._load_png_segmentation_mask(class_mask_filepath)
    )
    instance_mask = (
        self._load_mat_segmentation_mask(instance_mask_filepath, key="GTinst")
        if instance_mask_filepath.endswith(".mat")
        else self._load_png_segmentation_mask(instance_mask_filepath)
    )
    return {
        **super(SegmentationExampleBuilder, self).build(paths, example_id),
        "segmentation/class": class_mask,
        "segmentation/instance": instance_mask,
    }


class SplitConfig(object):
  def __init__(self, name, id_generator, example_builder):
    self.name = name
    self.id_generator = id_generator
    self.example_builder = example_builder


class VocConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Voc."""

  def __init__(self, homepage, citation, filenames=None, splits=[], **kwargs):
    self.homepage = homepage
    self.citation = citation
    self.filenames = filenames
    self.splits = splits
    super(VocConfig, self).__init__(
        # Version history:
        # 5.0.0: Added Segmentation BuildConfigs.
        # 4.0.0: Added BuildConfig and 2012 version support, deprecate Voc2007.
        # 3.0.0: S3 with new hashing function (different shuffle).
        # 2.0.0: S3 (new shuffling, sharding and slicing mechanism).
        version=tfds.core.Version("5.0.0"),
        **kwargs)

  @property
  def features(self):
    return tfds.features.FeaturesDict({
        "image": tfds.features.Image(shape=(None, None, 3), dtype=tf.uint8),
        "image/filename": tfds.features.Text(),
    })


class VocDetectionConfig(VocConfig):
  @property
  def features(self):
    return tfds.features.FeaturesDict({
        **super(VocDetectionConfig, self).features,
        "objects": tfds.features.Sequence({
            "label": tfds.features.ClassLabel(names=_VOC_LABELS),
            "bbox": tfds.features.BBoxFeature(),
            "pose": tfds.features.ClassLabel(names=_VOC_POSES),
            "is_truncated": tf.bool,
            "is_difficult": tf.bool,
        }),
        "labels": tfds.features.Sequence(
            tfds.features.ClassLabel(names=_VOC_LABELS)),
        "labels_no_difficult": tfds.features.Sequence(
            tfds.features.ClassLabel(names=_VOC_LABELS)),
    })


class VocSegmentationConfig(VocDetectionConfig):
  @property
  def features(self):
    return tfds.features.FeaturesDict({
        **super(VocSegmentationConfig, self).features,
        "segmentation/class": tfds.features.Image(shape=(None, None, 1), dtype=tf.uint8),
        "segmentation/instance": tfds.features.Image(shape=(None, None, 1), dtype=tf.uint8),
    })


class Voc(tfds.core.GeneratorBasedBuilder):
  """Pascal VOC 2007 or 2012."""

  BUILDER_CONFIGS = [
      VocDetectionConfig(
          name="2007",
          description=_VOC_DETECTION_DESCRIPTION.format(
              year=2007, num_images=9963, num_objects=24640),
          homepage=_VOC_URL.format(year="2007"),
          citation=_VOC_CITATION.format(year="2007"),
          filenames={
              "trainval": os.path.join(_VOC_DATA_URL, "VOCtrainval_06-Nov-2007.tar"),
              "test": os.path.join(_VOC_DATA_URL, "VOCtest_06-Nov-2007.tar"),
          },
          splits=[
              SplitConfig(
                  name=tfds.Split.TRAIN,
                  id_generator=IDGenerator("trainval", "VOCdevkit/VOC2007/ImageSets/Main/train.txt"),
                  example_builder=DetectionExampleBuilder(year=2007, filename="trainval")),
              SplitConfig(
                  name=tfds.Split.VALIDATION,
                  id_generator=IDGenerator("trainval", "VOCdevkit/VOC2007/ImageSets/Main/val.txt"),
                  example_builder=DetectionExampleBuilder(year=2007, filename="trainval")),
              SplitConfig(
                  name=tfds.Split.TEST,
                  id_generator=IDGenerator("test", "VOCdevkit/VOC2007/ImageSets/Main/test.txt"),
                  example_builder=DetectionExampleBuilder(year=2007, filename="test")),
          ],
      ),
      VocDetectionConfig(
          name="2012",
          description=_VOC_DETECTION_DESCRIPTION.format(
              year=2012, num_images=11540, num_objects=27450),
          homepage=_VOC_URL.format(year="2012"),
          citation=_VOC_CITATION.format(year="2012"),
          filenames={
              "trainval": os.path.join(_VOC_DATA_URL, "VOCtrainval_11-May-2012.tar"),
              "test": os.path.join(_VOC_DATA_URL, "VOC2012test.tar"),
          },
          splits=[
              SplitConfig(
                  name=tfds.Split.TRAIN,
                  id_generator=IDGenerator("trainval", "VOCdevkit/VOC2012/ImageSets/Main/train.txt"),
                  example_builder=DetectionExampleBuilder(year=2012, filename="trainval")),
              SplitConfig(
                  name=tfds.Split.VALIDATION,
                  id_generator=IDGenerator("trainval", "VOCdevkit/VOC2012/ImageSets/Main/val.txt"),
                  example_builder=DetectionExampleBuilder(year=2012, filename="trainval")),
              SplitConfig(
                  name=tfds.Split.TEST,
                  id_generator=IDGenerator("test", "VOCdevkit/VOC2012/ImageSets/Main/test.txt"),
                  # We pass `False` as the `annotation_path` because VOC2012 has
                  # no test annotations (`None` is used for the default).
                  example_builder=DetectionExampleBuilder(year=2012, filename="test", annotation_path=False)),
          ],
      ),
      VocSegmentationConfig(
          name="2007-segmentation",
          description=_VOC_SEGMENTATION_DESCRIPTION_2007,
          homepage=_VOC_URL.format(year=2007),
          citation=_VOC_CITATION.format(year=2007),
          filenames={
              "trainval": os.path.join(_VOC_DATA_URL, "VOCtrainval_06-Nov-2007.tar"),
              "test": os.path.join(_VOC_DATA_URL, "VOCtest_06-Nov-2007.tar"),
          },
          splits=[
              SplitConfig(
                  name=tfds.Split.TRAIN,
                  id_generator=IDGenerator("trainval", "VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt"),
                  example_builder=SegmentationExampleBuilder(year=2007, filename="trainval")),
              SplitConfig(
                  name=tfds.Split.VALIDATION,
                  id_generator=IDGenerator("trainval", "VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt"),
                  example_builder=SegmentationExampleBuilder(year=2007, filename="trainval")),
              SplitConfig(
                  name=tfds.Split.TEST,
                  id_generator=IDGenerator("test", "VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt"),
                  example_builder=SegmentationExampleBuilder(year=2007, filename="test")),
          ],
      ),
      VocSegmentationConfig(
          name="2012-segmentation",
          description=_VOC_SEGMENTATION_DESCRIPTION_2012,
          homepage=_VOC_URL.format(year="2012"),
          citation=_VOC_CITATION.format(year="2012") + _SBD_CITATION,
          filenames={
              "trainval": os.path.join(_VOC_DATA_URL, "VOCtrainval_11-May-2012.tar"),
              "test": os.path.join(_VOC_DATA_URL, "VOC2012test.tar"),
              "sbd": os.path.join(_SBD_DATA_URL, "semantic_contours/benchmark.tgz"),
          },
          splits=[
              SplitConfig(
                  name=tfds.Split.TRAIN,
                  id_generator=IDGenerator("trainval", "VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"),
                  example_builder=SegmentationExampleBuilder(year=2012, filename="trainval"),
              ),
              SplitConfig(
                  name=tfds.Split.VALIDATION,
                  id_generator=IDGenerator("trainval", "VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"),
                  example_builder=SegmentationExampleBuilder(year=2012, filename="trainval"),
              ),
              SplitConfig(
                  name="sbd_train",
                  id_generator=IncludeExcludeIDGenerator(
                      # The images of 'sbd_train' are all those contained in the
                      # SBD train split, excluding any included in the VOC2012
                      # train or validation split.
                      include_generators=[IDGenerator("sbd", "benchmark_RELEASE/dataset/train.txt")],
                      exclude_generators=[
                          IDGenerator("trainval", "VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"),
                          IDGenerator("trainval", "VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"),
                      ],
                  ),
                  example_builder=SegmentationExampleBuilder(
                      year=2012,
                      filename="trainval",
                      class_mask_path=VocPath("sbd", "benchmark_RELEASE/dataset/cls/{}.mat"),
                      instance_mask_path=VocPath("sbd", "benchmark_RELEASE/dataset/inst/{}.mat")),
              ),
              SplitConfig(
                  name="sbd_validation",
                  id_generator=IncludeExcludeIDGenerator(
                      # The images of 'sbd_validation' are all those contained
                      # in the SBD validation split, excluding any included in
                      # the VOC2012 train or validation split.
                      include_generators=[IDGenerator("sbd", "benchmark_RELEASE/dataset/val.txt")],
                      exclude_generators=[
                          IDGenerator("trainval", "VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"),
                          IDGenerator("trainval", "VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"),
                      ],
                  ),
                  example_builder=SegmentationExampleBuilder(
                      year=2012,
                      filename="trainval",
                      class_mask_path=VocPath("sbd", "benchmark_RELEASE/dataset/cls/{}.mat"),
                      instance_mask_path=VocPath("sbd", "benchmark_RELEASE/dataset/inst/{}.mat")),
              ),
          ],
      ),
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=self.builder_config.description,
        features=self.builder_config.features,
        homepage=self.builder_config.homepage,
        citation=self.builder_config.citation,
    )

  def _split_generators(self, dl_manager):
    paths = dl_manager.download_and_extract(self.builder_config.filenames)

    return [
        tfds.core.SplitGenerator(
            name=split.name,
            gen_kwargs={
                "id_generator": split.id_generator,
                "example_builder": split.example_builder,
                "paths": paths,
            },
        )
        for split in self.builder_config.splits
    ]

  def _generate_examples(self, paths, id_generator, example_builder):
    """Yields examples."""

    for example_id in id_generator.generate(paths):
      yield example_id, example_builder.build(paths, example_id)
