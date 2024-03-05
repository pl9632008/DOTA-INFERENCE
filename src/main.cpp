#include "rotated.h"


int main(){

    std::shared_ptr<Rotatation> rotatation = std::make_shared<Rotatation>();

    std::string path = "/home/ubuntu/wjd/rotate/models/yolov8s-obb.engine";
    std::string images_path = "/home/ubuntu/wjd/rotate/images";
    rotatation->loadEngine(path);
    rotatation->totalInference(images_path);


}