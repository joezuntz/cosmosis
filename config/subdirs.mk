define process_subdirs
	@for d in $(SUBDIRS); \
	do \
	  (cd $$d && $(MAKE) $@) || exit $$?; \
	done
endef

.PHONY: all clean test

all::
	$(process_subdirs)

test:
	$(process_subdirs)

clean:
	$(process_subdirs)

install:
  $(process_subbdirs)

