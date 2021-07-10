// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_DNN_FACE_CORE_HPP_
#define _OPENCV_DNN_FACE_CORE_HPP_

#include <opencv2/core.hpp>

/** @defgroup dnn_face DNN-based face detection and recognition
 */

namespace cv
{
namespace dnn_face
{
/** 
 */
class CV_EXPORTS_W DNNFaceDetector
{
    public:
        virtual ~DNNFaceDetector() {};
        CV_WRAP virtual Mat forward(const Mat& image) = 0;

        CV_WRAP static Ptr<DNNFaceDetector> create(const String& onnx_path,
                                           const int input_width,
                                           const int input_height,
                                           const float score_threshold = 0.9,
                                           const float nms_threshold = 0.3,
                                           const int top_k = 5000,
                                           const int backend_id = 0,
                                           const int target_id = 0);
};

} // namespace dnn_face
} // namespace cv

#endif
