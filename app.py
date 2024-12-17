from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
import numpy as np

# Khởi tạo Flask app
app = Flask(__name__)

# Load model đã chuyển sang TorchScript
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.jit.load('resnet_model_scripted.pt')
model.to(device)
model.eval()

# Chuyển đổi ảnh về dạng tensor
def transform(img, img_size=(224, 224)):
    img = img.resize(img_size)
    img = np.array(img)[..., :3]  # Chỉ lấy 3 kênh màu (RGB)
    img = torch.tensor(img).permute(2, 0, 1).float()  # Chuyển sang (C, H, W)
    normalized_img = img / 255  # Chuẩn hóa ảnh về [0, 1]
    return normalized_img.unsqueeze(0).to(device)  # Thêm batch dimension

# Định nghĩa route chính
@app.route('/')
def home():
    return render_template('index.html')

# Route cho dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    # Kiểm tra xem file có được gửi lên không
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Đọc ảnh và chuyển đổi
        img = Image.open(file)
        img_tensor = transform(img)
        
        # Chạy dự đoán
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            label = {0: 'dew', 1: 'fogsmog', 2: 'frost', 3: 'glaze', 4: 'hail',
                     5: 'lightning', 6: 'rain', 7: 'rainbow', 8: 'rime', 9: 'sandstorm', 10: 'snow'}
            prediction = label[predicted.item()]
        
        # Trả về kết quả dự đoán dưới dạng JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
