#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include <opencv2/face_det.hpp>

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage " << argv[0] << ": "
                  << "<onnx_path> "
                  << "<image>\n";
        std::cerr << "Download the face detection model at https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx\n";
        return -1;
    }

    String onnx_path = argv[1];
    String image_path = argv[2];
    Mat image = imread(image_path);

    float score_thresh = 0.9;
    float nms_thresh = 0.3;
    int top_k = 5000;

    // Initialize DNNFaceDetector
    dnn_face::DNNFaceDetector faceDetector(onnx_path);
    vector<dnn_face::Face> faces = faceDetector.detect(image, score_thresh, nms_thresh, top_k);

    // Visualize results
    const int thickness = 2;
    for (size_t i = 0; i < faces.size(); i++)
    {
        // Print results
        cout << "Face " << i << faces[i].box_tlwh << " " << faces[i].score << "\n";

        // Draw bounding box
        rectangle(image, faces[i].box_tlwh, Scalar(0, 255, 0), thickness);
        // Draw landmarks
        circle(image, faces[i].landmarks.right_eye,    2, Scalar(255,   0,   0), thickness);
        circle(image, faces[i].landmarks.left_eye,     2, Scalar(  0,   0, 255), thickness);
        circle(image, faces[i].landmarks.nose_tip,     2, Scalar(  0, 255,   0), thickness);
        circle(image, faces[i].landmarks.mouth_right,  2, Scalar(255,   0, 255), thickness);
        circle(image, faces[i].landmarks.mouth_left,   2, Scalar(  0, 255, 255), thickness);
    }

    try
    {
        // Save result image
        std::cout << "Saved to result.jpg\n";
        imwrite("result.jpg", image);
        // Display result image
        namedWindow(image_path, WINDOW_AUTOSIZE);
        imshow(image_path, image);
        waitKey(0);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    return 0;
}