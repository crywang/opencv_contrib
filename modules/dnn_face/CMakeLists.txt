set(the_description "Face Regconition using CNNs")


ocv_define_module(dnn_face
    opencv_core opencv_imgproc opencv_dnn
    OPTIONAL opencv_highgui opencv_imgcodecs  # samples
)
