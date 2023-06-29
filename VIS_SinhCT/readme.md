# Khởi tạo một object Formula_generator
<pre><code>```
from generator import Formula_generator
vis = Formula_generator(
    data: pd.DataFrame,
    generation_method: int = 0,
    required_fields: list[str] = [],
    multiple_cycles: bool = False,
    investment_method: int = 0,
    interest_rate: float = 1.00,
    measurement_method: int = 0,
    target: float = 1.0
)
```
</code></pre>

trong đó:
* data
* generation_method:
    - 0: Duyệt tất cả công thức (vét cạn toàn phần)
    - 1: Chỉ duyệt các công thức có các cụm con cùng cấu trúc (vét cạn nửa vời)

* required_fields: là một list các tên trường mà trong công thức sinh ra bắt buộc phải có các trường đó

* multiple_cycles: boolean, sinh cho nhiều chu kì hay không. Nếu là False thì chỉ sinh cho chu kì cuối cùng, người lại sẽ sinh số chu kì là N, tính từ chu kì cuối cùng trở về trước. N do người dùng nhập vào

* investment_method:
    - 0: Đầu tư công ty có value là value max của chu kì đầu tư, nếu có nhiều công ty có cùng value là value max thì không đầu tư.
    - 1: Đầu tư các công ty có value của năm đầu tư và năm trước đó đều vượt ngưỡng cho trước. Hoặc, đầu tư tất cả các công ty vượt ngưỡng nếu năm trước đó không có công ty nào vượt ngưỡng. Các trường hợp khác không đầu tư. Cách đầu tư này hiện chưa thể đánh giá theo cách 2 và 3.

* interest_rate: lãi suất tiền gửi trong trường hợp không đầu tư

* measurement_method:
    - 0: Geomean
    - 1: Harmean
    - 2: Geomean và độ chênh lệch giữa geomean và geo_limit. Độ chênh lệch do người dùng nhập vào.
    - 3: Harmean và độ chênh lệch giữa harmean và har_limit. Độ chênh lệch do người dùng nhập vào.

* target: profit (được đánh giá theo measurement_method) tối thiểu khi lưu công thức

# Các đối số bổ sung
* Khi generation_method là 0, cần bổ sung đối số numerator_condition (boolean). Khi là True, các công thức sinh ra với các cụm con có số toán hạng dưới mẫu không lớn hơn số toán hạng trên tử.

* Khi multiple_cycles là True, cần bổ sung đối số number_cycle, quy định số chu kì thực hiện sinh công thức, bắt đầu từ chu kì cuối cùng trở dần về trước.

* Khi investment_method là 1, cần bổ sung đối số max_loop, quy định số lần lặp tối đa để tìm ngưỡng tối ưu cho công thức.

* Khi measurement_method là 2 hoặc 3, cần bổ sung đối số diff_p_p_lim, quy định mức độ chênh lệch tối thiểu giữa profit và profit_limit của công thức được lưu.

# Chạy sinh công thức
<pre><code>```
vis.generate_formula(
    path: str,
    num_f_per_file: int=1000,
    num_f_target: int=1000000)
```
</code></pre>

trong đó:
* path: đường dẫn lưu công thức được sinh ra
* num_f_per_file: Số công thức tối đa trong một file lưu
* num_f_target: Số công thức tối đa được sinh ra trong một lần chạy

# Lưu lịch sử
Trong trường hợp ngắt quá trình chạy bằng tay (KeyboardInterrupt) thì cần chạy hàm lưu lịch sử:
<pre><code>```
vis.save_history()
```
</code></pre>