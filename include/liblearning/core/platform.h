#ifndef PLATFORM_H_
#define PLATFORM_H_

class platform
{
private:

	platform(void);
	~platform(void);

public:

	static void init();
	static void finalize();
};

#endif

