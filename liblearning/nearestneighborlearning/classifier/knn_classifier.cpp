#include <liblearning/nearestneighborlearning/classifier/knn_classifier.h>
#include <liblearning/util/algo_util.h>

#include <algorithm>

using namespace Eigen;

knn_classifier::knn_classifier(const supervised_dataset & train_, int k_):train(train_),k(k_)
{

}


knn_classifier::~knn_classifier(void)
{
}

#include <liblearning/util/Eigen_util.h>

double knn_classifier::test(const supervised_dataset & test)
{

	 MatrixXd dist = sqdist(train.get_data(),test.get_data());


	 vector<int> test_label(test.get_sample_num());

	 for (int i = 0;i<dist.cols();i++)
	 {
		 VectorXd cur_dist= dist.col(i);

		 vector<unsigned int> index =  nth_element_index(cur_dist.data(),cur_dist.data()+k,cur_dist.data()+cur_dist.size());

		 vector<int> class_labels(k);

		 std::transform(index.begin(),index.end(),class_labels.begin(),
			 [&](unsigned int n)-> int
			  {
				  return train.get_label()[n];
			  }
		 );

		 vector<int> elem_nums(k);
		 std::transform(class_labels.begin(),class_labels.end(),elem_nums.begin(),
			 [&class_labels](int n)-> int
			  {
				  return count(class_labels.begin(),class_labels.end(),n);
			  }
		 );

		 auto max_pos = std::max_element(elem_nums.begin(),elem_nums.end());

		 test_label[i] = class_labels[max_pos - elem_nums.begin()];
	 }

	 int error = 0;

	 for (int i =0; i<test.get_sample_num();i++)
	 {
		 error += (test_label[i] == test.get_label()[i]);
	 }

	 return double(error)/test.get_sample_num();
     
}