#include <liblearning/core/platform.h>


platform::platform(void)
{
}


platform::~platform(void)
{
}

#include <liblearning/util/math_util.h>
#include <liblearning/core/serialize.h>
#include <liblearning/core/dataset.h>
#include <liblearning/core/supervised_dataset.h>
#include <liblearning/core/cross_validation_pair.h>
#include <liblearning/core/train_test_pair.h>
#include <liblearning/core/experiment_datasets.h>

void platform::init()
{
	init_math_utils();
	
	camp::Class::declare<dataset>("dataset").constructor0();
	camp::Class::declare<supervised_dataset>("supervised_dataset").constructor0().base<dataset>();

	camp::Class::declare<cross_validation_pair>("cross_validation_pair").constructor0();
	camp::Class::declare<train_test_pair>("train_test_pair").constructor0();
	camp::Class::declare<experiment_datasets>("experiment_datasets").constructor0();


}
void platform::finalize()
{
	finish_math_utils();
}