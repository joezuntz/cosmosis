define process_subdirs
	@for d in $(SUBDIRS); \
	do \
	  (cd $$d && make $@); \
	done
endef

.PHONY: all

all: 
	$(process_subdirs)

test:
	$(process_subdirs)

clean:
	$(process_subdirs)

