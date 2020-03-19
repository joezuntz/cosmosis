#ifdef __cplusplus
extern "C" {
#endif
extern const char* BLOCK_LOG_READ;
extern const char* BLOCK_LOG_WRITE;
extern const char* BLOCK_LOG_READ_FAIL;
extern const char* BLOCK_LOG_WRITE_FAIL;
extern const char* BLOCK_LOG_READ_DEFAULT;
extern const char* BLOCK_LOG_REPLACE;
extern const char* BLOCK_LOG_REPLACE_FAIL;
extern const char* BLOCK_LOG_CLEAR;
extern const char* BLOCK_LOG_DELETE;
extern const char* BLOCK_LOG_START_MODULE;
extern const char* BLOCK_LOG_COPY;

#ifdef __cplusplus
}
#endif



#ifdef __cplusplus
#include <string>
#include <typeindex>
#include <typeinfo>
namespace cosmosis
{
  typedef std::tuple<std::string, std::string, std::string, std::type_index> log_entry;
}
#endif



