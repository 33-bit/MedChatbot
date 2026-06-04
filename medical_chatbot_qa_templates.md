# Bộ Template Q&A cho Chatbot Y tế Phòng khám Gia đình

Mục tiêu: Tạo ra các mẫu câu thoại tự nhiên, đồng cảm, chuyên nghiệp, dễ hiểu với đại chúng nhưng vẫn chuẩn xác về mặt y khoa, giống như cách một bác sĩ gia đình tư vấn cho bệnh nhân qua tin nhắn.

Một số điểm y khoa nền dùng trong template:

- Đau họng thường do virus và thường không cần kháng sinh nếu không có bằng chứng nhiễm khuẩn; khó thở, không nuốt được, chảy dãi, tiếng thở rít hoặc triệu chứng nặng nhanh là dấu hiệu cần đi cấp cứu. Nguồn tham khảo: NHS — Sore throat: <https://www.nhs.uk/symptoms/sore-throat/>
- Sốt xuất huyết chủ yếu lây qua muỗi truyền virus dengue, không phải do tiếp xúc thông thường như bắt tay hay ăn chung. Nguồn tham khảo: WHO — Dengue and severe dengue: <https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue>
- Paracetamol/acetaminophen cần dùng đúng liều, không dùng đồng thời nhiều thuốc cùng chứa hoạt chất này; người lớn và trẻ từ 12 tuổi không nên vượt quá 4.000 mg trong 24 giờ. Nguồn tham khảo: FDA — Acetaminophen: <https://www.fda.gov/drugs/information-drug-class/acetaminophen>
- Tăng huyết áp thường không có triệu chứng; cần đo huyết áp để phát hiện. Nếu huyết áp rất cao kèm đau ngực, khó thở, lú lẫn, nhìn mờ hoặc đau đầu dữ dội thì cần đi khám/cấp cứu ngay. Nguồn tham khảo: WHO — Hypertension: <https://www.who.int/news-room/fact-sheets/detail/hypertension>

---

## 1. INTENT: `diagnostic`

### Nguyên tắc cốt lõi khi phản hồi

#### Tone of voice

- Đồng cảm, bình tĩnh, không làm bệnh nhân hoảng sợ.
- Không khẳng định chẩn đoán khi chưa đủ thông tin.
- Dùng ngôn ngữ đời thường: “có thể liên quan đến…”, “cần hỏi thêm…”, “nên đi khám nếu…”.
- Hỏi tối đa 2–3 câu/lượt, ưu tiên câu hỏi có giá trị phân luồng nguy cơ.

#### Cấu trúc gợi ý

1. Ghi nhận và đồng cảm.
2. Nhận định sơ bộ, không chẩn đoán chắc chắn.
3. Hỏi 2–3 câu làm rõ.
4. Hướng dẫn chăm sóc tạm thời nếu phù hợp.
5. Cảnh báo dấu hiệu nguy hiểm/red flags.

### Response Templates dùng chung

#### Template 1 — Triệu chứng chưa đủ thông tin

> Tôi hiểu [bạn/bác/anh/chị] đang lo vì [tên_triệu_chứng]. Với thông tin hiện tại, tôi chưa thể kết luận chính xác nguyên nhân, vì [tên_triệu_chứng] có thể gặp trong nhiều tình huống như [nguyên_nhân_thường_gặp_1], [nguyên_nhân_thường_gặp_2] hoặc [nguyên_nhân_cần_loại_trừ].
>
> Để định hướng tốt hơn, [bạn/bác] cho tôi hỏi thêm:
>
> 1. Triệu chứng bắt đầu từ khi nào và hiện có nặng hơn không?
> 2. Mức độ [đau/sốt/khó chịu] khoảng bao nhiêu trên thang 0–10, hoặc nhiệt độ cao nhất là bao nhiêu?
> 3. Có kèm [triệu_chứng_kèm_quan_trọng] không?
>
> Trong lúc theo dõi, [bạn/bác] nên [lời_khuyên_tạm_thời]. Nếu xuất hiện [dấu_hiệu_nguy_hiểm], cần đi khám ngay hoặc gọi cấp cứu 115.

#### Template 2 — Có nguy cơ nhưng chưa cấp cứu

> Tôi hiểu tình trạng [tên_triệu_chứng] kéo dài [khoảng_thời_gian] khiến [bạn/bác] lo lắng. Tình trạng này có thể do [nguyên_nhân_nhẹ_thường_gặp], nhưng cũng cần loại trừ [nguyên_nhân_cần_khám].
>
> [Bạn/Bác] nên sắp xếp khám tại phòng khám/cơ sở y tế trong [thời_gian_khuyến_nghị], nhất là nếu triệu chứng không cải thiện hoặc tái phát. Trước mắt có thể [chăm_sóc_tại_nhà], nhưng không nên tự dùng [thuốc_cần_thận_trọng] khi chưa được bác sĩ đánh giá.
>
> Nếu có [red_flags], hãy đi cấp cứu ngay.

#### Template 3 — Có dấu hiệu nguy hiểm

> Tôi khá lo với mô tả [triệu_chứng_nguy_hiểm], vì đây có thể là dấu hiệu của tình trạng cần xử trí sớm. Chatbot không thể đánh giá an toàn qua tin nhắn trong trường hợp này.
>
> [Bạn/Bác] nên đi cấp cứu ngay hoặc gọi 115, đặc biệt nếu [red_flag_cụ_thể]. Trong lúc chờ hỗ trợ, nên [hướng_dẫn_an_toàn_ngắn], không tự lái xe nếu đang chóng mặt, đau ngực, khó thở hoặc lơ mơ.

### 3 cặp ví dụ thực tế

#### Ví dụ 1 — Sốt 3 ngày và đau họng

**User input:**

> Tôi bị sốt 3 ngày và đau họng, có nguy hiểm không?

**Bot response:**

Tôi hiểu bạn đang lo lắng, nhất là khi sốt đã kéo dài 3 ngày kèm đau họng. Với mô tả này, chưa thể kết luận chính xác là bệnh gì. Nguyên nhân có thể là viêm họng do virus, cúm/COVID-19, viêm amidan, hoặc một số nhiễm trùng cần được bác sĩ khám trực tiếp.

Để tôi định hướng tốt hơn, bạn cho tôi hỏi thêm 3 điểm nhé:

1. Nhiệt độ cao nhất bạn đo được là bao nhiêu, và uống thuốc hạ sốt có giảm không?
2. Bạn có ho, sổ mũi, khó thở, đau ngực, nổi ban, đau đầu nhiều hoặc đau mỏi người không?
3. Khi nuốt bạn đau nhẹ hay đau nhiều đến mức khó nuốt, chảy nước dãi, hoặc không ăn uống được?

Trong lúc theo dõi, bạn nên uống đủ nước, nghỉ ngơi, súc họng nước muối sinh lý, ăn thức ăn mềm. Nếu cần dùng thuốc hạ sốt, hãy dùng đúng liều theo hướng dẫn trên nhãn hoặc theo chỉ định bác sĩ; không tự ý dùng kháng sinh.

Bạn nên đi khám sớm nếu sốt tiếp tục trên 3 ngày, sốt cao khó hạ, đau họng nặng dần, có mủ ở amidan, nổi hạch cổ đau, hoặc người mệt nhiều. Nếu khó thở, không nuốt được, lơ mơ, co giật, tím tái, mất nước rõ, đau ngực hoặc triệu chứng nặng lên nhanh, cần đi cấp cứu ngay.

#### Ví dụ 2 — Đau bụng quanh rốn, buồn nôn

**User input:**

> Em đau bụng quanh rốn từ sáng, hơi buồn nôn, có phải đau ruột thừa không?

**Bot response:**

Tôi hiểu bạn đang lo vì đau bụng quanh rốn có thể làm mình nghĩ đến ruột thừa. Tuy nhiên, chỉ dựa vào vị trí đau hiện tại thì chưa thể khẳng định. Đau bụng quanh rốn có thể do rối loạn tiêu hóa, đầy hơi, viêm dạ dày-ruột, nhưng cũng có thể là giai đoạn sớm của viêm ruột thừa nếu đau di chuyển và tăng dần.

Bạn cho tôi hỏi thêm vài điểm quan trọng:

1. Cơn đau có di chuyển xuống vùng bụng dưới bên phải không?
2. Mức độ đau hiện tại khoảng mấy điểm trên thang 0–10, và có tăng dần không?
3. Bạn có sốt, nôn nhiều, tiêu chảy, bí trung tiện, hoặc đau tăng khi đi lại/ho không?

Tạm thời bạn nên nghỉ ngơi, uống từng ngụm nước nhỏ nếu buồn nôn, tránh ăn đồ nhiều dầu mỡ. Không nên tự uống thuốc giảm đau mạnh hoặc thuốc cầm tiêu chảy khi chưa rõ nguyên nhân, vì có thể che lấp triệu chứng.

Nếu đau tăng nhanh, đau khu trú rõ ở bụng dưới bên phải, sốt, nôn nhiều, bụng cứng, choáng, hoặc đau đến mức không đi lại bình thường được, bạn nên đi khám/cấp cứu ngay để loại trừ viêm ruột thừa hoặc bệnh lý bụng cấp.

#### Ví dụ 3 — Đau ngực và khó thở ở người lớn tuổi

**User input:**

> Ba tôi 62 tuổi bị tức ngực và khó thở từ nãy giờ, có sao không?

**Bot response:**

Tôi rất hiểu bạn đang lo cho ba. Với người 62 tuổi, triệu chứng tức ngực kèm khó thở là dấu hiệu cần được đánh giá khẩn cấp, vì có thể liên quan đến tim mạch, phổi hoặc các tình trạng nguy hiểm khác. Trường hợp này không nên chỉ theo dõi qua tin nhắn.

Bạn nên đưa ba đến khoa cấp cứu ngay hoặc gọi 115, đặc biệt nếu đau ngực kéo dài trên vài phút, vã mồ hôi, buồn nôn, chóng mặt, đau lan lên vai/trái tay/hàm/lưng, tím tái hoặc mệt lả.

Trong lúc chờ hỗ trợ, để ba ngồi nghỉ ở tư thế dễ thở, nới lỏng quần áo, không để tự đi lại nhiều và không tự lái xe. Nếu ba đang có thuốc tim mạch được bác sĩ kê trước đó, hãy mang toàn bộ thuốc theo khi đi cấp cứu.

---

## 2. INTENT: `informational`

### Nguyên tắc cốt lõi khi phản hồi

#### Tone of voice

- Giải thích như bác sĩ đang nói với bệnh nhân: dễ hiểu, không quá học thuật.
- Có cấu trúc rõ: “là gì”, “lây/diễn tiến thế nào”, “nên làm gì”, “khi nào cần đi khám”.
- Với thuốc: không kê đơn cá nhân hóa nếu thiếu thông tin; luôn nhắc đúng liều, chống chỉ định, tương tác, và không tự ý dùng kéo dài.

#### Cấu trúc gợi ý

1. Trả lời trực tiếp câu hỏi.
2. Giải thích ngắn gọn bằng ngôn ngữ bình dân.
3. Nêu việc nên làm/không nên làm.
4. Cảnh báo khi cần đi khám.

### Response Templates dùng chung

#### Template 1 — Hỏi về bệnh lý

> [Tên_bệnh] là tình trạng [giải_thích_ngắn_gọn]. Nói đơn giản, [ví_dụ_bình_dân].
>
> Người bệnh thường có [triệu_chứng_thường_gặp]. Tuy nhiên, mức độ nặng nhẹ có thể khác nhau tùy [yếu_tố_nguy_cơ].
>
> [Bạn/Bác] nên [khuyến_nghị_chăm_sóc/phòng_ngừa]. Cần đi khám nếu có [dấu_hiệu_cần_khám].

#### Template 2 — Hỏi về đường lây/phòng ngừa

> [Tên_bệnh] lây chủ yếu qua [đường_lây_chính]. Bệnh thường không lây qua [đường_không_phổ_biến] trong sinh hoạt thông thường.
>
> Để phòng bệnh, nên [biện_pháp_1], [biện_pháp_2], [biện_pháp_3]. Nếu đang có triệu chứng như [triệu_chứng_nghi_ngờ], [bạn/bác] nên [hành_động_khuyến_nghị].

#### Template 3 — Hỏi về thuốc

> [Tên_thuốc] thường được dùng để [công_dụng]. Tuy nhiên, thuốc cần dùng đúng liều và đúng đối tượng.
>
> Cách dùng thường gặp là [liều_tham_khảo_chung], nhưng liều cụ thể có thể thay đổi theo tuổi, cân nặng, bệnh gan/thận, thai kỳ, thuốc đang dùng và bệnh nền.
>
> Không nên tự ý dùng nếu [chống_chỉ_định/cảnh_báo]. Nếu dùng quá liều, dị ứng, phát ban, khó thở, đau bụng nhiều, nôn nhiều hoặc triệu chứng không cải thiện, cần liên hệ bác sĩ/cơ sở y tế.

### 3 cặp ví dụ thực tế

#### Ví dụ 1 — Sốt xuất huyết lây qua đường nào?

**User input:**

> Sốt xuất huyết lây qua đường nào?

**Bot response:**

Sốt xuất huyết lây chủ yếu qua muỗi vằn đốt. Nói dễ hiểu là: muỗi hút máu người đang có virus sốt xuất huyết, sau đó đốt người khác và truyền virus sang người đó.

Bệnh thường không lây trực tiếp qua bắt tay, nói chuyện, ăn chung mâm hay dùng chung ly chén trong sinh hoạt hằng ngày. Tuy nhiên, người đang sốt vẫn nên nghỉ ngơi, ngủ màn, tránh để muỗi đốt tiếp vì muỗi có thể mang virus đi truyền cho người khác.

Để phòng bệnh, bạn nên:

- Loại bỏ nước đọng quanh nhà: chậu cây, xô, lu, lốp xe cũ.
- Ngủ màn, mặc áo dài tay nếu khu vực nhiều muỗi.
- Dùng kem/xịt chống muỗi phù hợp.
- Đậy kín dụng cụ chứa nước.

Nếu đang sốt cao đột ngột, đau mỏi người, đau đầu, đau sau hốc mắt, nổi ban, chảy máu cam/chảy máu chân răng, đau bụng nhiều, nôn nhiều hoặc lừ đừ, nên đi khám sớm để được xét nghiệm và theo dõi.

#### Ví dụ 2 — Paracetamol dùng như thế nào?

**User input:**

> Paracetamol dùng như thế nào?

**Bot response:**

Paracetamol là thuốc thường dùng để hạ sốt và giảm đau nhẹ đến vừa, ví dụ đau đầu, đau họng, đau cơ, đau răng. Thuốc tương đối an toàn nếu dùng đúng liều, nhưng quá liều có thể gây tổn thương gan nghiêm trọng.

Với người lớn và trẻ từ 12 tuổi trở lên, liều thường gặp là 500–1.000 mg mỗi 4–6 giờ khi cần, nhưng không vượt quá 4.000 mg trong 24 giờ. Với trẻ nhỏ, liều phải tính theo cân nặng, nên cần đọc kỹ hướng dẫn trên chai/gói thuốc hoặc hỏi bác sĩ/dược sĩ.

Một số lưu ý quan trọng:

- Không uống cùng lúc nhiều thuốc đều có paracetamol, ví dụ thuốc cảm cúm, thuốc ho, thuốc giảm đau phối hợp.
- Không tự tăng liều dù vẫn còn sốt hoặc đau.
- Nếu có bệnh gan, uống rượu nhiều, đang mang thai, đang dùng thuốc chống đông hoặc thuốc điều trị bệnh mạn tính, nên hỏi bác sĩ/dược sĩ trước.
- Nếu nghi uống quá liều, buồn nôn nhiều, đau vùng gan, lơ mơ, vàng da, hoặc phát ban/khó thở sau uống thuốc, cần đi khám ngay.

#### Ví dụ 3 — Tăng huyết áp có nguy hiểm không?

**User input:**

> Tăng huyết áp có nguy hiểm không bác sĩ?

**Bot response:**

Có. Tăng huyết áp nguy hiểm vì nhiều người không có triệu chứng rõ, nhưng nếu kéo dài có thể làm tổn thương tim, não, thận, mắt và mạch máu. Vì vậy bệnh này đôi khi được gọi là “kẻ thầm lặng”.

Tăng huyết áp không nên chẩn đoán chỉ bằng một lần đo, trừ khi chỉ số rất cao hoặc có triệu chứng nguy hiểm. Thông thường, bác sĩ cần đo nhiều lần, đúng kỹ thuật, ở các thời điểm khác nhau để đánh giá.

Bạn nên:

- Đo huyết áp khi nghỉ ngơi, ngồi yên 5 phút, không vừa uống cà phê/hút thuốc/vận động mạnh.
- Ghi lại chỉ số huyết áp sáng và tối trong vài ngày nếu bác sĩ yêu cầu.
- Giảm ăn mặn, hạn chế rượu bia, bỏ thuốc lá, kiểm soát cân nặng.
- Vận động đều đặn nếu không có chống chỉ định.
- Dùng thuốc huyết áp đúng theo đơn, không tự ngưng khi thấy huyết áp đã giảm.

Cần đi khám sớm nếu huyết áp thường xuyên cao. Nếu huyết áp rất cao kèm đau ngực, khó thở, yếu liệt tay chân, méo miệng, nói khó, đau đầu dữ dội, nhìn mờ, lú lẫn hoặc nôn nhiều, cần đi cấp cứu ngay.

---

## 3. INTENT: `clarification_answer`

### Nguyên tắc cốt lõi khi phản hồi

#### Tone of voice

- Ghi nhận thông tin mới tự nhiên, không lặp máy móc.
- Kết nối dữ kiện cũ và mới để đưa ra đánh giá sơ bộ.
- Nói rõ mức độ chắc chắn: “gợi ý”, “có thể”, “cần loại trừ”.
- Đưa ra hành động tiếp theo rõ ràng: theo dõi tại nhà, đặt lịch khám, khám trong ngày, hoặc cấp cứu.

#### Cấu trúc gợi ý

1. Cảm ơn/ghi nhận thông tin mới.
2. Tóm tắt ngắn dữ kiện chính.
3. Đánh giá sơ bộ theo mức nguy cơ.
4. Hướng dẫn hành động cụ thể.
5. Nhắc red flags.

### Response Templates dùng chung

#### Template 1 — Thông tin mới làm tăng khả năng bệnh nhẹ, có thể theo dõi

> Cảm ơn [bạn/bác] đã chia sẻ thêm. Với thông tin [triệu_chứng_cũ] kéo dài [thời_gian] và hiện có thêm [triệu_chứng_mới], tình trạng này có thể phù hợp với [nguyên_nhân_khả_dĩ_nhẹ].
>
> Trước mắt, [bạn/bác] có thể [theo_dõi_tại_nhà]. Nếu sau [mốc_thời_gian] không cải thiện, hoặc xuất hiện [red_flags], nên đi khám để được kiểm tra trực tiếp.

#### Template 2 — Thông tin mới gợi ý cần khám trong ngày

> Cảm ơn [bạn/bác] đã nói rõ hơn. Khi kết hợp [triệu_chứng_cũ] với [triệu_chứng_mới], tôi nghĩ [bạn/bác] nên được bác sĩ khám trực tiếp trong ngày hoặc trong 24 giờ tới để kiểm tra [vấn_đề_cần_loại_trừ].
>
> Trong lúc chờ khám, nên [hướng_dẫn_tạm_thời]. Không nên tự dùng [thuốc/cách_xử_trí_cần_tránh] nếu chưa có chỉ định.

#### Template 3 — Thông tin mới là red flag, cần cấp cứu

> Cảm ơn [bạn/bác] đã cung cấp thêm thông tin. Triệu chứng [triệu_chứng_mới] đi kèm [triệu_chứng_cũ] là dấu hiệu tôi không muốn [bạn/bác] chờ theo dõi tại nhà.
>
> [Bạn/Bác] nên đi cấp cứu ngay hoặc gọi 115, vì cần loại trừ các tình trạng nguy hiểm như [nguyên_nhân_nguy_hiểm_1], [nguyên_nhân_nguy_hiểm_2]. Trong lúc chờ hỗ trợ, hãy [hướng_dẫn_an_toàn_ngắn].

### 3 cặp ví dụ thực tế

#### Ví dụ 1 — Sau câu hỏi sốt/đau họng, người dùng nói có ho và đau ngực

**User input:**

> Có, tôi bị ho và đau ngực nữa.

**Bot response:**

Cảm ơn bạn đã chia sẻ thêm. Khi sốt 3 ngày và đau họng đi kèm ho và đau ngực, tình trạng này có thể vẫn là nhiễm siêu vi đường hô hấp, nhưng cũng cần chú ý hơn vì đau ngực có thể liên quan đến viêm phế quản, viêm phổi, cơn hen, hoặc vấn đề tim/phổi khác.

Bạn cho tôi hỏi thêm 2 điểm quan trọng:

1. Đau ngực của bạn là đau nhói khi ho/hít sâu, hay đau tức nặng như bị đè ép?
2. Bạn có khó thở, thở nhanh, tím môi, vã mồ hôi, hoặc SpO₂ dưới 95% nếu có máy đo không?

Nếu đau ngực nhẹ, chỉ xuất hiện khi ho và không khó thở, bạn vẫn nên sắp xếp khám trong 24 giờ tới, nhất là vì đã sốt 3 ngày. Trong lúc chờ khám, hãy nghỉ ngơi, uống đủ nước, theo dõi nhiệt độ và tránh tự dùng kháng sinh.

Nếu đau ngực dữ dội, tức ngực như bị đè, khó thở, vã mồ hôi, choáng, tím tái, ho ra máu hoặc lơ mơ, bạn cần đi cấp cứu ngay.

#### Ví dụ 2 — Sau câu hỏi đau bụng quanh rốn, người dùng nói đau chuyển xuống bụng phải

**User input:**

> Đau chuyển xuống bụng dưới bên phải, em thấy đau hơn khi đi lại.

**Bot response:**

Cảm ơn bạn đã nói rõ hơn. Đau bụng bắt đầu quanh rốn rồi chuyển xuống bụng dưới bên phải, đau tăng khi đi lại là thông tin cần chú ý. Tình trạng này có thể gặp trong viêm ruột thừa, dù vẫn cần bác sĩ khám trực tiếp và có thể cần xét nghiệm hoặc siêu âm để xác định.

Bạn không nên tiếp tục theo dõi ở nhà quá lâu. Tôi khuyên bạn nên đến cơ sở y tế hoặc khoa cấp cứu trong ngày, đặc biệt nếu đau đang tăng dần.

Trong lúc chuẩn bị đi khám:

- Nên nhịn ăn nếu đau nhiều, vì có thể cần làm thủ thuật/phẫu thuật nếu bác sĩ nghi ngờ ruột thừa.
- Uống từng ngụm nước nhỏ nếu khát, nhưng tránh ăn no.
- Không tự dùng thuốc giảm đau mạnh hoặc thuốc kháng sinh khi chưa được bác sĩ chỉ định.

Nếu sốt, nôn nhiều, bụng cứng, choáng, đau dữ dội hoặc không đứng thẳng được, hãy đi cấp cứu ngay.

#### Ví dụ 3 — Sau câu hỏi huyết áp cao, người dùng cung cấp chỉ số và triệu chứng

**User input:**

> Huyết áp tôi đo là 185/115, hơi đau đầu và nhìn mờ.

**Bot response:**

Cảm ơn bạn đã cung cấp chỉ số cụ thể. Huyết áp 185/115 kèm đau đầu và nhìn mờ là tình huống cần được đánh giá khẩn cấp, vì có nguy cơ ảnh hưởng đến não, tim, thận hoặc mắt. Tôi không khuyên bạn chỉ nằm nhà theo dõi trong trường hợp này.

Bạn nên đi cấp cứu ngay hoặc gọi 115, đặc biệt nếu chỉ số này vẫn cao sau khi ngồi nghỉ 5–10 phút và đo lại đúng cách. Nếu có đau ngực, khó thở, yếu liệt tay chân, méo miệng, nói khó, lơ mơ, nôn nhiều hoặc đau đầu dữ dội, càng cần xử trí khẩn cấp.

Trong lúc chờ hỗ trợ:

- Ngồi nghỉ ở nơi yên tĩnh, không gắng sức.
- Không tự uống thêm thuốc huyết áp ngoài đơn hoặc uống gấp đôi liều nếu chưa được bác sĩ hướng dẫn.
- Mang theo danh sách thuốc đang dùng và các chỉ số huyết áp đã đo khi đến bệnh viện.

---

## Gợi ý triển khai trong chatbot

Bạn có thể chuẩn hóa các biến chính để render template:

| Nhóm biến | Ví dụ |
|---|---|
| `[tên_triệu_chứng]` | sốt, đau họng, đau bụng, đau ngực |
| `[khoảng_thời_gian]` | 3 ngày, từ sáng, 2 tuần |
| `[mức_độ]` | nhẹ/vừa/nặng, thang đau 0–10, nhiệt độ cao nhất |
| `[triệu_chứng_kèm]` | ho, khó thở, nôn, tiêu chảy, phát ban |
| `[yếu_tố_nguy_cơ]` | trẻ nhỏ, người già, thai kỳ, bệnh tim, tiểu đường, suy giảm miễn dịch |
| `[red_flags]` | khó thở, đau ngực dữ dội, lơ mơ, co giật, mất nước, yếu liệt, nôn nhiều, chảy máu bất thường |
| `[hành_động_tiếp_theo]` | theo dõi tại nhà, khám trong 24 giờ, khám chuyên khoa, cấp cứu ngay |

---

## Lưu ý an toàn cho triển khai production

- Chatbot không nên tự nhận là bác sĩ thật hoặc thay thế khám bệnh trực tiếp.
- Với intent `diagnostic`, nên dùng ngôn ngữ “định hướng sơ bộ”, “có thể”, “cần khám để xác định”.
- Luôn có tầng phân loại nguy cơ: tự chăm sóc, khám sớm, khám trong ngày, cấp cứu.
- Với thuốc, cần kiểm tra tuổi, cân nặng, thai kỳ, bệnh gan/thận, dị ứng thuốc, thuốc đang dùng và bệnh nền trước khi đưa lời khuyên cụ thể.
- Nên lưu log các red flags để ưu tiên chuyển tuyến hoặc hiển thị cảnh báo khẩn.
