// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"

#include <algorithm>

namespace cv
{
namespace dnn_face
{

class DNNFaceDetectorImpl : public DNNFaceDetector
{
public:
    DNNFaceDetectorImpl(const String& onnx_path,
                        const int input_width,
                        const int input_height,
                        const float score_threshold = 0.9,
                        const float nms_threshold = 0.3,
                        const int top_k = 5000,
                        const int backend_id = 1,
                        const int target_id = 1)
    {
        net = dnn::readNet(onnx_path);
        CV_Assert(!net.empty());

        net.setPreferableBackend(backend_id);
        net.setPreferableTarget(target_id);

        img_w = input_width;
        img_h = input_height;

        scoreThreshold = score_threshold;
        nmsThreshold = nms_threshold;
        topK = top_k;

        generatePriors();
    }

    Mat forward(const Mat& image) override
    {
        // Build blob from input image
        Mat blob = dnn::blobFromImage(image);

        // Forward
        std::vector<String> output_names = { "loc", "conf", "iou" };
        std::vector<Mat> output_blobs;
        net.setInput(blob);
        net.forward(output_blobs, output_names);

        // Post process
        Mat faces = postProcess(output_blobs);
        return faces;
    }
private:
    void generatePriors()
    {
        // Calculate shapes of different scales according to the shape of input image
        Size feature_map_2nd = {
            int(int((img_w+1)/2)/2), int(int((img_h+1)/2)/2)
        };
        Size feature_map_3rd = {
            int(feature_map_2nd.width/2), int(feature_map_2nd.height/2)
        };
        Size feature_map_4th = {
            int(feature_map_3rd.width/2), int(feature_map_3rd.height/2)
        };
        Size feature_map_5th = {
            int(feature_map_4th.width/2), int(feature_map_4th.height/2)
        };
        Size feature_map_6th = {
            int(feature_map_5th.width/2), int(feature_map_5th.height/2)
        };

        std::vector<Size> feature_map_sizes;
        feature_map_sizes.push_back(feature_map_3rd);
        feature_map_sizes.push_back(feature_map_4th);
        feature_map_sizes.push_back(feature_map_5th);
        feature_map_sizes.push_back(feature_map_6th);

        // Fixed params for generating priors
        const std::vector<std::vector<float>> min_sizes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}
        };
        const std::vector<int> steps = { 8, 16, 32, 64 };

        // Generate priors
        for (size_t i = 0; i < feature_map_sizes.size(); ++i)
        {
            Size feature_map_size = feature_map_sizes[i];
            std::vector<float> min_size = min_sizes[i];

            for (int _h = 0; _h < feature_map_size.height; ++_h)
            {
                for (int _w = 0; _w < feature_map_size.width; ++_w)
                {
                    for (size_t j = 0; j < min_size.size(); ++j)
                    {
                        float s_kx = min_size[j] / img_w;
                        float s_ky = min_size[j] / img_h;

                        float cx = (_w + 0.5) * steps[i] / img_w;
                        float cy = (_h + 0.5) * steps[i] / img_h;

                        Rect2f prior = { cx, cy, s_kx, s_ky };
                        priors.push_back(prior);
                    }
                }
            }
        }
    }

    Mat postProcess(const std::vector<Mat>& output_blobs)
    {
        // Extract from output_blobs
        Mat loc = output_blobs[0];
        Mat conf = output_blobs[1];
        Mat iou = output_blobs[2];

        // Decode from deltas and priors
        const std::vector<float> variance = {0.1, 0.2};
        Mat faces;
        float* loc_v = (float*)(loc.data);
        float* conf_v = (float*)(conf.data);
        float* iou_v = (float*)(iou.data);
        for (size_t i = 0; i < priors.size(); ++i) {
            // (tl_x, tl_y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score)
            // 'tl': top left point of the bounding box
            // 're': right eye, 'le': left eye
            // 'nt':  nose tip
            // 'rcm': right corner of mouth, 'lcm': left corner of mouth
            Mat face(1, 15, CV_32FC1);

            // Get score
            float clsScore = conf_v[i*2+1];
            float iouScore = iou_v[i];
            // Clamp
            if (iouScore < 0.f) {
                iouScore = 0.f;
            }
            else if (iouScore > 1.f) {
                iouScore = 1.f;
            }
            float score = std::sqrt(clsScore * iouScore);
            face.at<float>(0, 14) = score;

            // Get bounding box
            float cx = (priors[i].x + loc_v[i*14+0] * variance[0] * priors[i].width)  * img_w;
            float cy = (priors[i].y + loc_v[i*14+1] * variance[0] * priors[i].height) * img_h;
            float w  = priors[i].width  * exp(loc_v[i*14+2] * variance[0]) * img_w;
            float h  = priors[i].height * exp(loc_v[i*14+3] * variance[1]) * img_h;
            float x1 = cx - w / 2;
            float y1 = cy - h / 2;
            face.at<float>(0, 0) = x1;
            face.at<float>(0, 1) = y1;
            face.at<float>(0, 2) = w;
            face.at<float>(0, 3) = h;

            // Get landmarks
            face.at<float>(0, 4) = (priors[i].x + loc_v[i*14+ 4] * variance[0] * priors[i].width)  * img_w;  // right eye, x
            face.at<float>(0, 5) = (priors[i].y + loc_v[i*14+ 5] * variance[0] * priors[i].height) * img_h;  // right eye, y
            face.at<float>(0, 6) = (priors[i].x + loc_v[i*14+ 6] * variance[0] * priors[i].width)  * img_w;  // left eye, x
            face.at<float>(0, 7) = (priors[i].y + loc_v[i*14+ 7] * variance[0] * priors[i].height) * img_h;  // left eye, y
            face.at<float>(0, 8) = (priors[i].x + loc_v[i*14+ 8] * variance[0] * priors[i].width)  * img_w;  // nose tip, x
            face.at<float>(0, 9) = (priors[i].y + loc_v[i*14+ 9] * variance[0] * priors[i].height) * img_h;  // nose tip, y
            face.at<float>(0, 10) = (priors[i].x + loc_v[i*14+10] * variance[0] * priors[i].width)  * img_w; // right corner of mouth, x
            face.at<float>(0, 11) = (priors[i].y + loc_v[i*14+11] * variance[0] * priors[i].height) * img_h; // right corner of mouth, y
            face.at<float>(0, 12) = (priors[i].x + loc_v[i*14+12] * variance[0] * priors[i].width)  * img_w; // left corner of mouth, x
            face.at<float>(0, 13) = (priors[i].y + loc_v[i*14+13] * variance[0] * priors[i].height) * img_h; // left corner of mouth, y

            faces.push_back(face);
        }

        if (faces.rows > 1)
        {
            // Retrieve boxes and scores
            std::vector<Rect2i> faceBoxes;
            std::vector<float> faceScores;
            for (int rIdx = 0; rIdx < faces.rows; rIdx++)
            {
                faceBoxes.push_back(Rect2i(faces.at<float>(rIdx, 0),
                                           faces.at<float>(rIdx, 1),
                                           faces.at<float>(rIdx, 2),
                                           faces.at<float>(rIdx, 3)));
                faceScores.push_back(faces.at<float>(rIdx, 14));
            }

            std::vector<int> keepIdx;
            dnn::NMSBoxes(faceBoxes, faceScores, scoreThreshold, nmsThreshold, keepIdx, 1.f, topK);

            // Get NMS results
            Mat nms_faces;
            for (int idx: keepIdx)
            {
                nms_faces.push_back(faces.row(idx));
            }
            return nms_faces;
        }
        else
        {
            return faces;
        }
    }
private:
    dnn::Net net;

    int img_w;
    int img_h;
    float scoreThreshold;
    float nmsThreshold;
    int topK;

    std::vector<Rect2f> priors;
};

Ptr<DNNFaceDetector> DNNFaceDetector::create(const String& onnx_path,
                                             const int input_width,
                                             const int input_height,
                                             const float score_threshold,
                                             const float nms_threshold,
                                             const int top_k,
                                             const int backend_id,
                                             const int target_id)
{
    return makePtr<DNNFaceDetectorImpl>(onnx_path, input_width, input_height, score_threshold, nms_threshold, top_k, backend_id, target_id);
}

} // namespace dnn_face
} // namespace cv


