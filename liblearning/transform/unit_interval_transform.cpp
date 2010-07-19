#include <liblearning/transform/unit_interval_transform.h>


using namespace Eigen;

unit_interval_transform::unit_interval_transform(const dataset & train)
{
	min_elem = train.get_data().minCoeff();
	max_elem = train.get_data().maxCoeff();
}

unit_interval_transform::unit_interval_transform(double min_elem_, double max_elem):min_elem(min_elem_),max_elem(max_elem)
{


}

unit_interval_transform::~unit_interval_transform(void)
{
}


shared_ptr<dataset> unit_interval_transform::apply(const dataset & data_set)
{

	MatrixXd data = data_set.get_data();
	data = (data.array()- min_elem)/(max_elem-min_elem);
	return data_set.clone_update_data(data);
}