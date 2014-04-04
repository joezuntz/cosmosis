void   __wmap_options_MOD_wmap_init_options();

void __attribute__((constructor)) ctor()
{
  __wmap_options_MOD_wmap_init_options();
}
