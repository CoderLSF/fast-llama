
#pragma once

#include <stdio.h>
#include <stdlib.h>

#ifndef ft_log_debug
#define ft_log_debug(format, ...) printf("DEBUG: [%s()] " format "\n", __func__, ##__VA_ARGS__)
#endif

#ifndef ft_log_error
#define ft_log_error(format, ...) printf("ERROR: [%s:%d] [%s()] " format "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#endif

#ifndef tf_log_debug
#define tf_log_debug(format, ...) printf("DEBUG: [%s()] " format "\n", __func__, ##__VA_ARGS__)
#endif

#ifndef tf_log_error
#define tf_log_error(format, ...) printf("ERROR: [%s:%d] [%s()] " format "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#endif


