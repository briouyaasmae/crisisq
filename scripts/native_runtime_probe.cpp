#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <string>
#include <cmath>
int main(int argc,char**argv){try{
 if(argc!=5)return 2;
 Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"CrisisQ diagnostic");Ort::SessionOptions opts;opts.SetIntraOpNumThreads(2);opts.SetInterOpNumThreads(1);
 std::string profile=argv[3];
 if(profile=="no_kleidiai"||profile=="no_quant_fusions")opts.AddConfigEntry("mlas.disable_kleidiai","1");
 if(profile=="no_quant_fusions")opts.AddConfigEntry("optimization.disable_specified_optimizers","DynamicQuantizeMatMulFusion,MatMulIntegerToFloatFusion");
 if(profile=="no_graph_optimization")opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
 Ort::Session session(env,argv[1],opts);auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
 std::ifstream input(argv[2]);std::ofstream output(argv[4]);output<<std::setprecision(17);std::string id;std::array<int64_t,2> shape{1,128};const char* inputs[]{"input_ids","attention_mask"};const char* outputs[]{"logits"};int count=0;
 std::cerr<<"Runtime "<<Ort::GetVersionString()<<" profile "<<profile<<std::endl;
 while(input>>id){std::vector<int64_t> ids(128),mask(128);for(auto&v:ids)input>>v;for(auto&v:mask)input>>v;if(!input)throw std::runtime_error("Bad input");
  std::array<Ort::Value,2> values{Ort::Value::CreateTensor<int64_t>(memory,ids.data(),ids.size(),shape.data(),2),Ort::Value::CreateTensor<int64_t>(memory,mask.data(),mask.size(),shape.data(),2)};
  auto logits=session.Run(Ort::RunOptions{nullptr},inputs,values.data(),2,outputs,1);auto p=logits[0].GetTensorData<float>();double a=p[0],b=p[1],m=std::max(a,b);double prob=std::exp(b-m)/(std::exp(a-m)+std::exp(b-m));output<<id<<"\t"<<prob<<"\n";count++;
 }
 std::cerr<<"Complete "<<count<<std::endl;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}return 0;}
