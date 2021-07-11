# Face Regconition using Convolutional Neural Networks

This module uses Convolutional Neural Networks for face detection and regconition.

## Dependencies
- OpenCV DNN Module

## Build

Run the following command to build this module with OpenCV:
```shell
cmake -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules <opencv_source_dir>
```

## Models

There are two models (ONNX format) pre-trained and required for this module:
- [Face Detection](https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx):
    - Size: 337KB
    - Results on WIDER Face Val set: 0.830(easy), 0.824(medium), 0.708(hard)
- [Face Recognition](https://drive.google.com/file/d/1ClK9WiB492c5OZFKveF3XiHCejoOxINW/view?usp=sharing)
    - Size: 36.9MB
    - Results:

    | Database | Accuracy | Threshold (normL2) | Threshold (cosine) |
    | -------- | -------- | ------------------ | ------------------ |
    | LFW      | 99.60%   | 1.272              | 0.363              |
    | CALFW    | 93.95%   | 1.320              | 0.340              |
    | CPLFW    | 91.05%   | 1.450              | 0.275              |
    | AgeDB-30 | 94.90%   | 1.446              | 0.277              |
    | CFP-FP   | 94.80%   | 1.571              | 0.212              |

## Examples

### Face Detection
```cpp
// Initialize DNNFaceDetector with onnx_path (cv::String)
cv::dnn_face::DNNFaceDetector faceDetector(onnx_path);

// Run face detection with given image (cv::Mat), score_thresh (float), nms_thresh (float) and top_k (int)
std::vector<dnn_face::Face> faces = faceDetector.detect(image, score_thresh, nms_thresh, top_k);
```

`dnn_face::Face` is a struct holding a `cv::Rect2i` for face bounding boxes (top-left coordinates, width, height), a struct (`dnn_face::Landmarks_5`) variable for landmarks and a detection score (`float`).
```cpp
typedef struct Face
{
    cv::Rect2i box_tlwh;
    dnn_face::Landmarks_5 landmarks;
    float score;
} Face;
```

`dnn_face::Landmarks_5` holds the coordinates of 5 landmarks, which are right and left eyes, nose tip, the right and left corners of the mouth.
```cpp
typedef struct Landmarks_5
{
    cv::Point2i right_eye;
    cv::Point2i left_eye;
    cv::Point2i nose_tip;
    cv::Point2i mouth_right;
    cv::Point2i mouth_left;
} Landmarks_5;
```

### Face Recognition

Following Face Detection, run codes below to extract face feature from a facial image.

```cpp
// Initialize DNNFaceRecognizer with onnx_path (cv::String)
cv::dnn_face::DNNFaceRecognizer faceRecognizer(onnx_path);

// Aligning and cropping facial image with landmarks (dnn_face::Landmarks_5) detected by dnn_face::DNNFaceDetector
Mat aligned_face = faceRecognizer.AlignCrop(image, faces[0].landmarks);

// Run feature extraction with given aligned_face (cv::Mat)
Mat feature = faceRecognizer.facefeature(aligned_face);
```

After obtaining face features *feature1* and *feature2* of two facial images, run codes below to calculate the identity discrepancy between the two faces.

```cpp
// Calculating the discrepancy between two face features by using cosine distance.
float cos_score = faceRecognizer.facematch(feature1, feature2, "cosine");
// Calculating the discrepancy between two face features by using normL2 distance.
float L2_score = faceRecognizer.facematch(feature1, feature2, "norml2");
```

For example, two faces have same identity if the cosine distance is greater than or equal to 0.34, or the normL2 distance is less than or equal to 1.32.
