c = get_config()

# use xelatex for better unicode + svg support
c.LatexExporter.latex_command = ['xelatex', '{filename}']

# enable svg to pdf conversion
c.SVG2PDFPreprocessor.from_format = 'svg'
c.SVG2PDFPreprocessor.to_format = 'pdf'
c.LatexExporter.preprocessors = ['nbconvert.preprocessors.SVG2PDFPreprocessor']