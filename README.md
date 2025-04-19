# dot-matrix-OCR

Template matching OCR for dot-matrix LCD displays

Examples of dot-matrix LCD displays: [1](https://www.winstar.com.tw/uploads/photos/character-lcd-display-modules/WH1602A.JPG) [2](https://cdn-reichelt.de/bilder/web/artikel_ws/A500%2FEAW204B-NLW.jpg?type=Product&) [3](https://www.displayvisions.us/fileadmin/images/header/Header_EA_W202-XDLG.jpg)

## Pipeline

Photograph of display ->

1. (Optional) Perspective correction
2. Crop to display area
3. Preprocessing
4. Character detection
5. Template matching

-> Structured output
