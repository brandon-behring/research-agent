# Makefile for research-agent
#
# Targets:
#   spec         render docs/REQUIREMENTS_SPEC.md → build/*.docx + build/*.pdf
#   spec-clean   remove the build/ directory
#
# Requires Pandoc (for both targets) and a LaTeX engine (xelatex by default)
# for the PDF target. On Debian/Ubuntu:
#     sudo apt install pandoc texlive-xetex texlive-fonts-recommended

.PHONY: spec spec-clean

BUILD_DIR := build
SPEC_SRC := docs/REQUIREMENTS_SPEC.md
SPEC_DOCX := $(BUILD_DIR)/research-agent-SRS.docx
SPEC_PDF := $(BUILD_DIR)/research-agent-SRS.pdf

spec: $(SPEC_DOCX) $(SPEC_PDF)

$(SPEC_DOCX): $(SPEC_SRC)
	@mkdir -p $(BUILD_DIR)
	pandoc $(SPEC_SRC) -o $@ --toc --toc-depth=3 --number-sections
	@echo "Built: $@"

$(SPEC_PDF): $(SPEC_SRC)
	@mkdir -p $(BUILD_DIR)
	pandoc $(SPEC_SRC) -o $@ --pdf-engine=xelatex --toc --toc-depth=3 --number-sections
	@echo "Built: $@"

spec-clean:
	rm -rf $(BUILD_DIR)
	@echo "Removed: $(BUILD_DIR)/"
