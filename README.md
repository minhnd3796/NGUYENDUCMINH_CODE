Họ và tên: Nguyễn Đức Minh
MSSV: 20146488


Download dữ liệu tại địa chỉ: 
https://drive.google.com/file/d/1y8ZVRNlYFoofoV4eO5v2-XB-nqDEO-l5/view?usp=sharing
https://vngms-my.sharepoint.com/personal/duytv_vng_com_vn/Documents/ISPRS_semantic_labeling_Vaihingen.zip?csf=1&e=7429fad505494e828182184105ff1b4d

Cài đặt anaconda phiên bản python 3, cài đặt các gói cần thiết (tensorflow, numpy, libtiff...), cuda, cần chạy trên GPU của NVIDIA.
Project được chia ra thành các phần:
Các file Batch_manager*.py là các file chứa phần quản lý batch giúp trích các mảng (batch_size, image_size, image_size, number_of_channels) cho quá trình huấn luyện.
File convert_to_tfrecords*.py chuyển tập dữ liệu về dạng tfrecords, và sử dụng queue cho quá trình huấn luyện, giúp quá trình huấn luyện nhanh và cần ít bộ nhớ hơn.
File convert_to_annotations.py chuyển groundtruth dưới dạng ảnh 3 kênh màu về dạng ảnh đen trắng có giá trị pixel trong khoảng [0,5]
File calculate_mean*.py tính giá trị trung bình của các kênh dữ liệu của tập dữ liệu tương ứng.
File data_reader*.py hỗ trợ việc load dữ liệu.
File fully_convnet*.py là các mô hình fcn-8s cho 2 tập dữ liệu với số lượng kênh màu khác nhau.
File fully_conv_densenet*.py là các mô hình FC-Densenet cho 2 tập với số kênh màu khác nhau.
File infer_little_image*.py suy diễn các ảnh trên tập validate và tập test cho mô hình fcn-8s với số lượng kênh màu và tập dữ liệu tương ứng.
File infer_little_image*.py suy diễn các ảnh trên tập validate và tập test cho mô hình FC-Densenet với số lượng kênh màu và tập dữ liệu tương ứng.
File infer_numpy*.py suy diễn ra mảng numpy kích thước (H,W,6) phục vụ cho quá trình ensemble.
File infer_ensemble*.py kết hợp các file numpy để tiến hành ensemble mô hình.
File layers_fc_densenet.py có các layers phụ trợ cho việc cài đặt FC-Densenet.
File sampling_image*.py lẫy mẫu dữ liệu từ tập huấn luyện với số lượng kênh màu khác nhau và tập dữ liệu tương ứng.
File tensor_utils.py gồm các hàm hỗ trợ cho quá trình cài đặt các lớp của các mạng.

Quy trình tiến hành huấn luyện và suy diễn:
- Download dữ liệu và giải nén.
- Chạy file convert_gt2annotations.py với tập dữ liệu tương ứng.
- Chạy các file sampling_image.py để lấy mẫu dữ liệu.
- Chạy các mô hình ở các file tương ứng với dữ liệu lấy mẫu. e.g. fully_convnets.py
- Cuối cùng, sử dụng file infer_little_image*.py để dự đoán ảnh trong tập dữ liệu tương ứng với tập ảnh lấy mẫu.
