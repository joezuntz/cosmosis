define process_subdirs
	@for d in $(SUBDIRS); \
	do \
	  (cd $$d && make $@); \
	done
endef
