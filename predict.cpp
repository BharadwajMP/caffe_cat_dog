#include<caffe/caffe.hpp>
#include<caffe/util/io.hpp>
#include<caffe/blob.hpp>
#include "/usr/local/include/caffe/layers/memory_data_layer.hpp"
using namespace caffe;

int main(){
    boost::shared_ptr<caffe::Net<float> > net;
    net.reset(new caffe::Net<float>("./caffe.prototxt", caffe::TEST));
    net->CopyTrainedLayersFrom("solver_iter_1000.caffemodel");
    Datum datum;
    ReadImageToDatum("cat.jpeg",1,&datum);
    MemoryDataLayer<float> *test = (MemoryDataLayer<float> *)(net->layer_by_name("data").get());

    Blob<float>* blob = new Blob<float>(1, datum.channels(), datum.height(), datum.width());
    BlobProto blob_proto;
    blob_proto.set_num(1);
    blob_proto.set_channels(datum.channels());
    blob_proto.set_height(datum.height());
    blob_proto.set_width(datum.width());
    const int data_size = datum.channels() * datum.height() * datum.width();
    int size_in_datum = std::max<int>(datum.data().size(),
        datum.float_data_size());
    for (int i = 0; i < size_in_datum; ++i) {
        blob_proto.add_data(0.);
    }
    const string& data = datum.data();
    if (data.size() != 0) {
        for (int i = 0; i < size_in_datum; ++i) {
            blob_proto.set_data(i, blob_proto.data(i) + (uint8_t)data[i]);
        }
    }
  
    blob->FromProto(blob_proto);
    vector<Blob<float>*> bottom;
    bottom.push_back(blob);
    float type = 0.0;
    test->Reset(bottom,type,1);
    const vector<Blob<float>*>& result =  net->Forward();
    float max = 0;
    float max_i = 0;
    for (int i = 0; i < 1000; ++i) {
        float value = result[0]->gpu_data()[i];
        if (max < value){
            max = value;
            max_i = i;
        }
    }

    std::cout<<"Predicted class = "<<max<<std::endl;
    return 0;
}