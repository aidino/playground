# Fairseq examples

## MMPT - VideoCLIP and VLM

Bạn vừa khám phá ra bộ công cụ tuyệt vời để hiểu video đa phương thức! Nó bao gồm 2 phương pháp mới nhất: VideoCLIP và VLM, cùng với những công cụ hiệu suất cao thường thiếu trong các bộ mã hiện có. Bộ công cụ này được thiết kế với các thành phần được tinh chỉnh hiệu suất, có thể thích ứng với nhiều khung khác nhau (ban đầu sử dụng fairseq).

* **VideoCLIP:** là mô hình học tương phản, dùng để chuyển giao zero-shot (không cần huấn luyện lại) cho các tác vụ như truy xuất thông tin, phân loại và gán nhãn chuỗi trong video.
    * **Giải thích:** VideoCLIP học bằng cách so sánh các cặp video và văn bản, từ đó hiểu được mối quan hệ giữa chúng. Nhờ vậy, nó có thể thực hiện các tác vụ mới mà không cần huấn luyện lại từ đầu.
* **VLM:** là mô hình ngôn ngữ với kỹ thuật che (mask), sử dụng một bộ mã hóa duy nhất với mô hình phương thức che (MMM) cho các tác vụ truy xuất, tạo văn bản và gán nhãn chuỗi.
    * **Giải thích:** VLM học bằng cách che đi một số phần của video hoặc văn bản, sau đó dự đoán phần bị che. Điều này giúp mô hình hiểu sâu hơn về nội dung và ngữ cảnh.

Tóm lại, bộ công cụ này cung cấp các phương pháp mạnh mẽ để phân tích và hiểu video đa phương thức, với khả năng ứng dụng cao trong nhiều lĩnh vực.

## adaptive_span - Adaptive Span

Adaptive Span là một cơ chế tự chú ý mới có thể tự học khoảng chú ý tối ưu của nó. Điều này cho phép chúng ta mở rộng đáng kể kích thước ngữ cảnh tối đa được sử dụng trong Transformer, đồng thời vẫn kiểm soát được lượng bộ nhớ và thời gian tính toán. Nó sử dụng kỹ thuật Truncated BPTT để huấn luyện, giống như trong transformerXL.

**Giải thích:**

* **Cơ chế tự chú ý (self-attention):** là một kỹ thuật trong AI cho phép mô hình tập trung vào các phần quan trọng của dữ liệu đầu vào. Trong Transformer, self-attention giúp mô hình hiểu được mối quan hệ giữa các từ trong một câu.
* **Khoảng chú ý (attention span):**  là phạm vi mà mô hình tập trung vào khi xử lý dữ liệu. Khoảng chú ý càng lớn, mô hình càng có thể xem xét nhiều thông tin hơn, nhưng đồng thời cũng tốn nhiều bộ nhớ và thời gian tính toán hơn.
* **Adaptive Span:**  giúp mô hình tự động điều chỉnh khoảng chú ý sao cho phù hợp với từng ngữ cảnh. Điều này giúp cải thiện hiệu suất của mô hình mà không làm tăng đáng kể tài nguyên cần thiết.
* **Truncated BPTT:**  là một kỹ thuật huấn luyện giúp giảm thiểu lượng bộ nhớ cần thiết khi huấn luyện các mô hình có chuỗi dài, như Transformer.

Adaptive Span được giới thiệu trong bài báo "Adaptive Attention Span in Transformers", đạt kết quả  ngôn ngữ học tiên tiến nhất tại thời điểm xuất bản.

Chúng tôi đã tái tạo kết quả của họ trong fairseq và giữ nguyên hầu hết code ban đầu. Bạn có thể tham khảo tệp sweep của họ nếu có bất kỳ sự kết hợp nào của siêu tham số không rõ ràng.

## attention_head_selection

**Paper: Pay Better Attention to Attention: Head Selection in Multilingual and Multi-Domain Sequence Modeling (Gong et al., 2021)**

https://arxiv.org/pdf/2106.10840.pdf

Bài báo "Pay Better Attention to Attention: Head Selection in Multilingual and Multi-Domain Sequence Modeling" (Gong et al., 2021) nghiên cứu các chiến lược lựa chọn **attention head** trong mô hình hóa chuỗi đa ngôn ngữ và đa lĩnh vực, bao gồm các tác vụ dịch thuật văn bản, nhận dạng giọng nói và dịch giọng nói.

**Giải thích:**

* **Attention head:**  Mỗi **attention head** trong mô hình Transformer học cách tập trung vào các phần khác nhau của dữ liệu đầu vào. Lựa chọn **attention head** phù hợp có thể giúp mô hình hoạt động hiệu quả hơn.
* **Multilingual and multi-domain sequence modeling:**  Xây dựng mô hình AI có thể xử lý nhiều ngôn ngữ hoặc nhiều lĩnh vực khác nhau (ví dụ: y tế, tài chính).

**Ví dụ:**

Bài báo đưa ra ví dụ về việc huấn luyện mô hình nhận dạng giọng nói đa ngôn ngữ/đa lĩnh vực. Bằng cách lựa chọn **attention head** hiệu quả, mô hình có thể cải thiện độ chính xác khi nhận dạng giọng nói từ nhiều ngôn ngữ hoặc trong nhiều ngữ cảnh khác nhau.

**Tóm lại:**

Bài báo đề xuất phương pháp cải thiện hiệu suất của mô hình Transformer bằng cách lựa chọn **attention head** phù hợp với từng tác vụ và dữ liệu.

## audio_nlp/nlu - End-to-end NLU

**Hiểu Ngôn ngữ Tự nhiên (NLU) từ đầu đến cuối**

NLU từ đầu đến cuối (End-to-end NLU) dự đoán ý định trực tiếp từ âm thanh bằng cách sử dụng một mô hình duy nhất. Phương pháp này hứa hẹn cải thiện hiệu suất của các hệ thống trợ lý bằng cách tận dụng thông tin âm thanh bị mất trong biểu diễn văn bản trung gian và ngăn chặn lỗi lan truyền từ Nhận dạng giọng nói tự động (ASR). Hơn nữa, việc có một mô hình thống nhất mang lại lợi thế về hiệu quả khi triển khai các hệ thống trợ lý trên thiết bị.

**Giải thích:**

* **End-to-end NLU:** Thay vì xử lý giọng nói theo từng bước riêng biệt (nhận dạng giọng nói thành văn bản, rồi phân tích ý nghĩa từ văn bản), end-to-end NLU sử dụng một mô hình duy nhất để trực tiếp phân tích ý nghĩa từ âm thanh.
* **Lợi ích:**
    * Tận dụng thông tin âm thanh (như ngữ điệu, cảm xúc) để hiểu ý định chính xác hơn.
    * Tránh lỗi lan truyền: Lỗi trong bước nhận dạng giọng nói (ASR) có thể ảnh hưởng đến kết quả phân tích ý nghĩa. End-to-end NLU loại bỏ bước trung gian này, giảm thiểu lỗi.
    * Hiệu quả: Một mô hình duy nhất giúp tiết kiệm tài nguyên và dễ dàng triển khai trên thiết bị.

Trang này phát hành mã để tái tạo kết quả trong STOP: Bộ dữ liệu cho Phân tích ngữ nghĩa hướng tác vụ bằng giọng nói.


## backtranslation

**Paper: Understanding Back-Translation at Scale (Edunov et al., 2018)**

Trang này bao gồm các mô hình được huấn luyện trước (pre-trained models) từ bài báo "Understanding Back-Translation at Scale" (Edunov et al., 2018).

**Giải thích:**

* **Back-translation:**  Là một kỹ thuật trong dịch máy, trong đó văn bản đích được dịch ngược lại sang ngôn ngữ nguồn. Kỹ thuật này giúp tạo thêm dữ liệu huấn luyện và cải thiện hiệu suất của mô hình dịch.
* **Understanding Back-Translation at Scale:**  Bài báo nghiên cứu hiệu quả của back-translation trên quy mô lớn, tức là với lượng dữ liệu huấn luyện khổng lồ.

Tóm lại, trang này cung cấp các mô hình dịch máy đã được huấn luyện trước bằng kỹ thuật back-translation trên quy mô lớn, dựa trên nghiên cứu của Edunov et al. (2018).

## bart
**Paper: BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension**

https://arxiv.org/abs/1910.13461


**Giới thiệu**

BART là một mô hình sequence-to-sequence được huấn luyện trước với mục tiêu **denoising**. Chúng tôi chứng minh rằng mục tiêu huấn luyện trước này mang tính tổng quát hơn và cho thấy rằng chúng tôi có thể đạt được kết quả tương đương RoBERTa trên SQuAD và GLUE và đạt được kết quả tiên tiến nhất (state-of-the-art) về tóm tắt văn bản (XSum, CNN dataset), trả lời câu hỏi dạng dài (ELI5) và tạo phản hồi hội thoại (ConvAI2). Xem bài báo liên quan để biết thêm chi tiết.

**Giải thích:**

* **Sequence-to-Sequence model:** Mô hình nhận đầu vào là một chuỗi và tạo ra đầu ra cũng là một chuỗi. Ví dụ: dịch máy (chuỗi đầu vào là câu tiếng Anh, chuỗi đầu ra là câu tiếng Việt), tóm tắt văn bản.
* **Denoising:** Kỹ thuật huấn luyện trước bằng cách làm nhiễu dữ liệu đầu vào (ví dụ: xóa từ, thay đổi thứ tự từ), sau đó yêu cầu mô hình khôi phục lại dữ liệu ban đầu.
* **BART:**  Kết hợp kiến trúc encoder-decoder của mô hình sequence-to-sequence với mục tiêu huấn luyện trước denoising.
* **Ưu điểm:** BART linh hoạt hơn các phương pháp huấn luyện trước khác, đạt kết quả tốt trên nhiều tác vụ ngôn ngữ khác nhau.

Tóm lại, BART là một mô hình mạnh mẽ trong xử lý ngôn ngữ tự nhiên, được huấn luyện trước bằng phương pháp denoising để đạt hiệu quả cao trong nhiều tác vụ như tạo ngôn ngữ, dịch thuật và hiểu văn bản.


## byte_level_bpe - Neural Machine Translation with Byte-Level Subwords

https://arxiv.org/abs/1909.03341

We provide an implementation of byte-level byte-pair encoding (BBPE), taking IWSLT 2017 Fr-En translation as example.

## camembert

**CamemBERT: a Tasty French Language Model**

CamemBERT is a pretrained language model trained on 138GB of French text based on RoBERTa.

Also available in github.com/huggingface/transformers.

## constrained_decoding

**(Vectorized) Lexically constrained decoding with dynamic beam allocation**

Trang này cung cấp hướng dẫn về cách sử dụng giải mã bị ràng buộc từ vựng (lexically constrained decoding) trong Fairseq. Fairseq triển khai mã được mô tả trong các bài báo sau:

* Fast Lexically Constrained Decoding With Dynamic Beam Allocation (Post & Vilar, 2018)
* Improved Lexically Constrained Decoding for Translation and Monolingual Rewriting (Hu et al., 2019)


**Giải thích:**

* **Lexically constrained decoding:** Kỹ thuật giải mã trong dịch máy, trong đó ta ép buộc mô hình phải đưa ra bản dịch chứa các từ hoặc cụm từ cho trước.
* **Dynamic beam allocation:** Kỹ thuật phân bổ chùm tia động, giúp tăng hiệu quả tìm kiếm trong quá trình giải mã.
* **Ứng dụng:**
    * Cải thiện chất lượng dịch thuật bằng cách đảm bảo bản dịch chứa các thuật ngữ quan trọng.
    * Hỗ trợ dịch thuật sáng tạo, ví dụ như viết thơ hoặc chơi chữ.

Ví dụ trên cho thấy cách sử dụng Fairseq để dịch câu tiếng Đức sang tiếng Anh, đồng thời ràng buộc bản dịch phải chứa các từ "hard" và "to influence".

## conv_seq2seq

**Convolutional Sequence to Sequence Learning (Gehring et al., 2017)**


Bài báo "Convolutional Sequence to Sequence Learning" đã giới thiệu một cách tiếp cận mới để xử lý chuỗi bằng cách sử dụng CNNs, mang lại hiệu quả vượt trội so với các phương pháp truyền thống dựa trên RNNs. Đây là một bước tiến quan trọng trong lĩnh vực xử lý ngôn ngữ tự nhiên và học máy.

## criss

**Cross-lingual Retrieval for Iterative Self-Supervised Training**

https://arxiv.org/pdf/2006.09526.pdf


**Giới thiệu**

CRISS là một phương pháp huấn luyện trước (pretraining) đa ngôn ngữ theo kiểu **sequence-to-sequence**, trong đó quá trình khai thác (mining) và huấn luyện (training) được áp dụng lặp đi lặp lại, cải thiện khả năng **căn chỉnh chéo ngôn ngữ** (cross-lingual alignment) và **dịch thuật** đồng thời.

**Giải thích:**

* **Cross-lingual Retrieval:**  Kỹ thuật truy xuất thông tin trên nhiều ngôn ngữ. Ví dụ: tìm kiếm tài liệu tiếng Việt tương ứng với một tài liệu tiếng Anh.
* **Iterative Self-Supervised Training:** Huấn luyện tự giám sát lặp đi lặp lại. Mô hình tự tạo ra dữ liệu huấn luyện từ dữ liệu chưa được gán nhãn, sau đó sử dụng dữ liệu này để cải thiện bản thân.
* **Sequence-to-Sequence:**  Kiến trúc mô hình AI thường dùng trong dịch máy, tóm tắt văn bản, nơi đầu vào và đầu ra đều là chuỗi.

**CRISS hoạt động như sau:**

1. **Khai thác dữ liệu:** Sử dụng kỹ thuật truy xuất chéo ngôn ngữ để tìm các cặp câu tương ứng giữa các ngôn ngữ khác nhau.
2. **Huấn luyện mô hình:**  Sử dụng các cặp câu này để huấn luyện mô hình sequence-to-sequence.
3. **Lặp lại:**  Quá trình khai thác và huấn luyện được lặp lại nhiều lần. Mô hình càng được huấn luyện, khả năng truy xuất chéo ngôn ngữ càng tốt, từ đó tạo ra dữ liệu huấn luyện chất lượng cao hơn, giúp cải thiện mô hình hơn nữa.

**Tóm lại:**

CRISS là một phương pháp huấn luyện trước mới, kết hợp truy xuất chéo ngôn ngữ và huấn luyện tự giám sát lặp đi lặp lại, giúp cải thiện hiệu quả của mô hình sequence-to-sequence trong các tác vụ đa ngôn ngữ như dịch thuật.

## cross_lingual_language_model

**Cross-Lingual Language Model Pre-training**

**Huấn luyện trước Mô hình Ngôn ngữ Xuyên Ngôn ngữ (Cross-Lingual Language Model Pre-training)**

Dưới đây là một số chi tiết về huấn luyện Mô hình Ngôn ngữ Xuyên Ngôn ngữ (Cross-Lingual Language Models - XLM) - tương tự như mô hình được trình bày trong Lample & Conneau, 2019 - trong Fairseq. Triển khai hiện tại chỉ hỗ trợ Mô hình Ngôn ngữ Che (Masked Language Model - MLM) từ bài báo trên.

**Giải thích:**

* **Cross-lingual Language Model (XLM):**  Là một loại mô hình ngôn ngữ được huấn luyện trên dữ liệu của nhiều ngôn ngữ khác nhau, cho phép mô hình học được các biểu diễn chung (shared representations) giữa các ngôn ngữ. Điều này giúp cải thiện hiệu suất của mô hình trên các tác vụ xử lý ngôn ngữ tự nhiên đa ngôn ngữ, ví dụ như dịch máy, phân tích cảm xúc đa ngôn ngữ.
* **Pre-training:**  Quá trình huấn luyện mô hình trên một lượng dữ liệu lớn trước khi tinh chỉnh (fine-tuning) cho một tác vụ cụ thể. 
* **Masked Language Model (MLM):**  Một kỹ thuật huấn luyện trước, trong đó mô hình được yêu cầu dự đoán các từ bị che (masked) trong một câu, dựa trên ngữ cảnh xung quanh.

**Tóm lại:**

Fairseq cung cấp triển khai cho việc huấn luyện trước mô hình ngôn ngữ xuyên ngôn ngữ (XLM) bằng kỹ thuật MLM.  XLM giúp mô hình học được các đặc trưng chung giữa các ngôn ngữ, từ đó cải thiện hiệu suất trên các tác vụ đa ngôn ngữ.


## data2vec

**data2vec 2.0**

**data2vec 2.0**

data2vec 2.0 cải thiện hiệu quả huấn luyện của thuật toán data2vec ban đầu. Chúng tôi thực hiện các cải tiến sau để tăng hiệu quả: chỉ chuyển các bước thời gian không bị che (unmasked timesteps) qua bộ mã hóa (encoder), sử dụng bộ giải mã tích chập (convolutional decoder) và sử dụng kỹ thuật che đa lớp (multimasking) để phân bổ chi phí tính toán của mô hình giáo viên (teacher model). Bạn có thể tìm thấy chi tiết trong bài báo "Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language" và bài đăng trên blog của chúng tôi.

**Giải thích:**

* **data2vec:**  Là một thuật toán học tự giám sát (self-supervised learning) được sử dụng để huấn luyện các mô hình biểu diễn dữ liệu (representation learning) cho nhiều loại dữ liệu khác nhau, bao gồm hình ảnh, giọng nói và ngôn ngữ.
* **Self-supervised learning:**  Một phương pháp huấn luyện mô hình AI mà không cần sử dụng dữ liệu có nhãn. Mô hình tự học từ chính dữ liệu đầu vào bằng cách tạo ra các nhiệm vụ giả (pretext tasks), ví dụ như dự đoán phần bị che của dữ liệu.
* **data2vec 2.0:**  Phiên bản cải tiến của data2vec, tập trung vào việc tăng hiệu quả huấn luyện.
* **Các cải tiến:**
    * **Chỉ chuyển các bước thời gian không bị che qua bộ mã hóa:** Giúp giảm lượng tính toán cần thiết.
    * **Sử dụng bộ giải mã tích chập:**  Tận dụng khả năng tính toán song song của mạng nơ-ron tích chập để tăng tốc độ huấn luyện.
    * **Sử dụng kỹ thuật che đa lớp:**  Cho phép mô hình học từ nhiều mức độ che khác nhau, từ đó tăng cường khả năng biểu diễn dữ liệu.

**Tóm lại:**

data2vec 2.0 là một phiên bản cải tiến của thuật toán data2vec, mang lại hiệu quả huấn luyện cao hơn đáng kể so với phiên bản trước đó. Các cải tiến tập trung vào việc giảm lượng tính toán và tăng tốc độ huấn luyện, giúp data2vec 2.0 trở thành một lựa chọn hấp dẫn cho việc huấn luyện các mô hình biểu diễn dữ liệu tự giám sát.

## discriminative_reranking_nmt

**Discriminative Reranking for Neural Machine Translation**

**Xếp hạng lại Phân biệt đối xử cho Dịch máy Thần kinh (Discriminative Reranking for Neural Machine Translation)**

Đây là tiêu đề của một bài báo khoa học về cải thiện chất lượng dịch máy. 

**Giải thích:**

* **Neural Machine Translation (NMT):**  Dịch máy thần kinh, sử dụng mạng nơ-ron nhân tạo để dịch ngôn ngữ.
* **Reranking:**  Kỹ thuật xếp hạng lại các bản dịch được tạo ra bởi mô hình NMT. Mô hình NMT thường tạo ra nhiều bản dịch khả thi, reranking giúp chọn ra bản dịch tốt nhất.
* **Discriminative:**  Phương pháp phân biệt đối xử, tập trung vào việc phân biệt giữa các bản dịch tốt và xấu.

**Cách thức hoạt động:**

1. **Tạo ra nhiều bản dịch:** Mô hình NMT ban đầu tạo ra một số lượng lớn các bản dịch khả thi.
2. **Trích xuất đặc trưng:**  Từ mỗi bản dịch, các đặc trưng (features) quan trọng được trích xuất, ví dụ: độ trôi chảy, độ chính xác về nghĩa, sự tương đồng với câu nguồn.
3. **Huấn luyện mô hình xếp hạng:**  Một mô hình học máy riêng biệt (thường là mô hình phân loại) được huấn luyện để phân biệt giữa các bản dịch tốt và xấu dựa trên các đặc trưng đã trích xuất.
4. **Xếp hạng lại:**  Mô hình xếp hạng được sử dụng để đánh giá và xếp hạng tất cả các bản dịch được tạo ra ở bước 1. Bản dịch có điểm số cao nhất sẽ được chọn là bản dịch cuối cùng.

**Lợi ích:**

* **Cải thiện độ chính xác:** Bằng cách xem xét nhiều bản dịch khả thi và sử dụng mô hình xếp hạng, kỹ thuật này giúp chọn ra bản dịch chính xác và tự nhiên hơn.
* **Linh hoạt:** Có thể kết hợp nhiều loại đặc trưng khác nhau để đánh giá bản dịch.

**Tóm lại:**

Bài báo này đề xuất một phương pháp để cải thiện chất lượng dịch máy bằng cách sử dụng kỹ thuật xếp hạng lại phân biệt đối xử. Phương pháp này giúp lựa chọn bản dịch tốt nhất từ nhiều bản dịch khả thi, mang lại kết quả dịch chính xác và tự nhiên hơn.


## emotion_conversion

**Textless speech emotion conversion using decomposed and discrete representations**

**Chuyển đổi Cảm xúc Giọng nói không cần Văn bản sử dụng Biểu diễn Phân rã và Rời rạc**

**Tóm tắt:**

Chuyển đổi cảm xúc giọng nói (Speech emotion conversion) là nhiệm vụ thay đổi cảm xúc nhận thức của một lời nói trong khi vẫn giữ nguyên nội dung từ vựng và danh tính người nói. Trong nghiên cứu này, chúng tôi coi vấn đề chuyển đổi cảm xúc như một nhiệm vụ dịch ngôn ngữ nói. Chúng tôi phân rã giọng nói thành các biểu diễn rời rạc và được gỡ rối, bao gồm các đơn vị nội dung, F0, người nói và cảm xúc. Đầu tiên, chúng tôi sửa đổi nội dung giọng nói bằng cách dịch các đơn vị nội dung sang cảm xúc mục tiêu, và sau đó dự đoán các đặc trưng ngữ điệu dựa trên các đơn vị này. Cuối cùng, dạng sóng giọng nói được tạo ra bằng cách đưa các biểu diễn được dự đoán vào bộ giải mã giọng nói thần kinh (neural vocoder). Mô hình như vậy cho phép chúng tôi vượt ra ngoài các thay đổi phổ và tham số của tín hiệu, và mô hình hóa các giọng nói phi ngôn ngữ, chẳng hạn như chèn tiếng cười, loại bỏ tiếng ngáp, v.v. Chúng tôi chứng minh một cách khách quan và chủ quan rằng phương pháp được đề xuất vượt trội so với các phương pháp cơ sở về cảm xúc nhận thức và chất lượng âm thanh. Chúng tôi đánh giá nghiêm ngặt tất cả các thành phần của một hệ thống phức tạp như vậy và kết luận với một phân tích mô hình mở rộng và nghiên cứu loại bỏ để nhấn mạnh tốt hơn các lựa chọn kiến trúc, điểm mạnh và điểm yếu của phương pháp được đề xuất. Các mẫu và mã sẽ được công khai tại liên kết sau: [https://speechbot.github.io/emotion](https://speechbot.github.io/emotion).

**Giải thích:**

* **Speech emotion conversion:**  Thay đổi cảm xúc của giọng nói (ví dụ: từ buồn sang vui) mà không làm thay đổi nội dung và người nói.
* **Decomposed and discrete representations:** Biểu diễn giọng nói dưới dạng các thành phần riêng biệt (nội dung, cao độ, người nói, cảm xúc) và rời rạc (có thể phân loại).
* **Spoken language translation:** Coi việc chuyển đổi cảm xúc như dịch từ ngôn ngữ "cảm xúc A" sang ngôn ngữ "cảm xúc B".
* **Neural vocoder:**  Mô hình AI tạo ra giọng nói từ các thông tin đầu vào (ví dụ: văn bản, đặc trưng âm thanh).

**Cách thức hoạt động:**

1. **Phân rã giọng nói:** Tách giọng nói thành các biểu diễn rời rạc (nội dung, F0, người nói, cảm xúc).
2. **Chuyển đổi nội dung:**  "Dịch" nội dung sang cảm xúc mục tiêu.
3. **Dự đoán ngữ điệu:**  Dự đoán các đặc trưng ngữ điệu dựa trên nội dung đã chuyển đổi.
4. **Tạo giọng nói:**  Kết hợp các biểu diễn để tạo ra giọng nói mới với cảm xúc mong muốn.

**Kết quả:**

Nghiên cứu cho thấy phương pháp này tạo ra giọng nói với cảm xúc tự nhiên hơn và chất lượng âm thanh tốt hơn so với các phương pháp trước đây.

**Tóm lại:**

Đây là một nghiên cứu tiên tiến trong lĩnh vực chuyển đổi cảm xúc giọng nói, sử dụng các kỹ thuật học sâu và biểu diễn rời rạc để tạo ra giọng nói với cảm xúc chân thực và chất lượng cao.


## fast_noisy_channel

**Language Models not just for Pre-training: Fast Online Neural Noisy Channel Modeling**

**Giới thiệu**

Yee và cộng sự (2019) giới thiệu một phương pháp mô hình kênh nhiễu (noisy channel modeling) đơn giản và hiệu quả cho dịch máy thần kinh (neural machine translation). Tuy nhiên, phương pháp giải mã trực tuyến kênh nhiễu được giới thiệu trong bài báo này quá chậm để áp dụng thực tế.

Để giải quyết vấn đề này, Bhosale và cộng sự (2020) giới thiệu 3 phép xấp xỉ đơn giản để làm cho phương pháp này rất nhanh và thực tế mà không làm giảm nhiều độ chính xác.

Tệp README này cung cấp hướng dẫn về cách chạy giải mã hoặc tạo trực tuyến với phương pháp mô hình kênh nhiễu, bao gồm các cách để làm cho nó rất nhanh mà không làm giảm nhiều độ chính xác.

**Mô hình Kênh Nhiễu (Noisy Channel Modeling)**

Yee và cộng sự (2019) áp dụng Quy tắc Bayes để dự đoán $P(y|x)$, xác suất của mục tiêu y với điều kiện nguồn x. $$P(y|x) = P(x|y) * P(y) / P(x)$$

* $P(x|y)$ dự đoán nguồn x với điều kiện mục tiêu y và được gọi là mô hình kênh (channel model)
* $P(y)$ là mô hình ngôn ngữ (language model) trên mục tiêu y
* $P(x)$ thường không được mô hình hóa vì nó là hằng số cho tất cả y.

Chúng tôi sử dụng mô hình Transformer để tham số hóa mô hình trực tiếp P(y|x), mô hình kênh P(x|y) và mô hình ngôn ngữ P(y).

Trong quá trình giải mã trực tuyến với tìm kiếm chùm tia (beam search), chúng tôi tạo ra K2 ứng cử viên hàng đầu cho mỗi chùm tia và chấm điểm chúng với tổ hợp tuyến tính sau của mô hình kênh, mô hình ngôn ngữ cũng như điểm số mô hình trực tiếp.

$$(1 / t) * log(P(y|x) + (1 / s) * ( λ1 * log(P(x|y)) + λ2 * log(P(y) ) )$$

* $t$ - Độ dài Tiền tố Mục tiêu (Target Prefix Length)
* $s$ - Độ dài Nguồn (Source Length)
* $λ1$ - Trọng số Mô hình Kênh (Channel Model Weight)
* $λ2$ - Trọng số Mô hình Ngôn ngữ (Language Model Weight)

Các ứng cử viên `beam_size` hàng đầu dựa trên điểm số kết hợp ở trên được chọn để tiếp tục các chùm tia trong tìm kiếm chùm tia. Trong tìm kiếm chùm tia chỉ với một mô hình trực tiếp, điểm số từ mô hình trực tiếp P(y|x) được sử dụng để chọn các ứng cử viên hàng đầu trong tìm kiếm chùm tia.

Khung này cung cấp một cách tuyệt vời để sử dụng các mô hình ngôn ngữ mục tiêu mạnh mẽ được huấn luyện trên một lượng lớn dữ liệu không được gắn nhãn. Mô hình ngôn ngữ có thể ưu tiên các mục tiêu không liên quan đến nguồn, vì vậy chúng tôi cũng cần một mô hình kênh có vai trò đảm bảo rằng mục tiêu được mô hình ngôn ngữ ưu tiên cũng dịch ngược lại thành nguồn.

**Huấn luyện Mô hình Dịch và Mô hình Ngôn ngữ**

Để huấn luyện mô hình Transformer trong fairseq cho dịch máy, hãy tham khảo hướng dẫn tại đây

Để huấn luyện mô hình Transformer trong fairseq cho mô hình hóa ngôn ngữ, hãy tham khảo hướng dẫn tại đây

**Tạo với Mô hình Ngôn ngữ cho dịch Đức-Anh với fairseq**

Dưới đây là hướng dẫn để tạo bằng cách sử dụng mô hình trực tiếp và mô hình ngôn ngữ phía mục tiêu.

**Lưu ý:**

* Tải xuống và cài đặt fairseq theo hướng dẫn tại đây
* Tiền xử lý và nhị phân hóa tập dữ liệu theo hướng dẫn trong phần Tiền xử lý Dữ liệu Kiểm tra (Test Data Preprocessing)


**Giải thích:**

Bài viết này tập trung vào việc sử dụng mô hình ngôn ngữ (language model) để cải thiện hiệu quả của dịch máy thần kinh (neural machine translation). Thay vì chỉ dựa vào mô hình dịch trực tiếp (direct model), phương pháp này kết hợp thêm mô hình ngôn ngữ và mô hình kênh (channel model) để tạo ra bản dịch tốt hơn. 

Mô hình kênh có vai trò đảm bảo rằng bản dịch được tạo ra không chỉ đúng ngữ pháp mà còn phù hợp với ngữ cảnh của câu nguồn.

Tuy nhiên, việc giải mã trực tuyến với mô hình kênh nhiễu ban đầu gặp vấn đề về tốc độ. Bài viết này giới thiệu các cải tiến để tăng tốc độ giải mã mà không làm giảm đáng kể độ chính xác.

**Keywords quan trọng:**

* **Noisy channel modeling:** Mô hình kênh nhiễu
* **Neural machine translation:** Dịch máy thần kinh
* **Language model:** Mô hình ngôn ngữ
* **Channel model:** Mô hình kênh
* **Beam search:** Tìm kiếm chùm tia
* **Online decoding:** Giải mã trực tuyến


Hy vọng bản dịch và giải thích này dễ hiểu. Nếu bạn có bất kỳ câu hỏi nào, đừng ngần ngại hỏi nhé!

## flores101

**Flores101: Large-Scale Multilingual Machine Translation**

Flores101 là một dự án nghiên cứu về dịch máy (Machine Translation) tập trung vào việc xây dựng một hệ thống dịch đa ngôn ngữ (Multilingual) với quy mô lớn (Large-Scale). 

**Giải thích:**

* **Flores101:** Tên của tập dữ liệu và hệ thống dịch máy. 
* **Large-Scale:**  Flores101 được huấn luyện trên một lượng dữ liệu khổng lồ, bao gồm 101 ngôn ngữ. Điều này giúp mô hình học được các biểu diễn chung (shared representations) giữa các ngôn ngữ, từ đó cải thiện hiệu suất dịch.
* **Multilingual Machine Translation:**  Khả năng dịch giữa nhiều cặp ngôn ngữ khác nhau. Ví dụ, Flores101 có thể dịch từ tiếng Anh sang tiếng Việt, từ tiếng Pháp sang tiếng Nhật, v.v.

**Mục tiêu của Flores101:**

* **Cải thiện chất lượng dịch:**  Flores101 hướng đến việc tạo ra các bản dịch chính xác và tự nhiên hơn, đặc biệt là đối với các cặp ngôn ngữ ít tài nguyên (low-resource languages).
* **Đánh giá hiệu quả dịch:**  Flores101 cung cấp một bộ dữ liệu đánh giá (evaluation dataset) đa ngôn ngữ, cho phép các nhà nghiên cứu so sánh hiệu suất của các mô hình dịch khác nhau.
* **Hỗ trợ nghiên cứu:**  Flores101 là một tài nguyên quý giá cho cộng đồng nghiên cứu dịch máy, giúp thúc đẩy sự phát triển của các hệ thống dịch đa ngôn ngữ hiệu quả hơn.


**Tóm lại:**

Flores101 là một dự án quan trọng trong lĩnh vực dịch máy, góp phần xây dựng các hệ thống dịch đa ngôn ngữ quy mô lớn, chất lượng cao và hỗ trợ nghiên cứu trong lĩnh vực này.

## fully_sharded_data_parallel

**Fully Sharded Data Parallel (FSDP)**

**Tổng quan**

Các nghiên cứu gần đây của Microsoft và Google đã chỉ ra rằng huấn luyện song song dữ liệu (data parallel training) có thể được thực hiện hiệu quả hơn đáng kể bằng cách chia sẻ các tham số mô hình (model parameters) và trạng thái trình tối ưu hóa (optimizer state) trên các worker song song dữ liệu. Những ý tưởng này được gói gọn trong trình bao bọc FullyShardedDataParallel (FSDP) mới do fairscale cung cấp.

**So sánh với PyTorch DDP:**

* FSDP tạo ra kết quả giống hệt như PyTorch DDP (nó vẫn là huấn luyện song song dữ liệu đồng bộ)
* FSDP chia sẻ các tham số (FP16 + FP32) và trạng thái trình tối ưu hóa trên các GPU song song dữ liệu
* FSDP nhanh hơn PyTorch DDP vì bước trình tối ưu hóa được chia sẻ và việc giao tiếp có thể được chồng chéo với bước chuyển tiếp (forward pass)
* FSDP cho phép huấn luyện mô hình 13 tỷ tham số trên 8 GPU và mô hình 175 tỷ tham số trên 128 GPU
* FSDP được hỗ trợ đầy đủ trong fairseq thông qua các đối số mới sau:
    * `--ddp-backend=fully_sharded`: bật chia sẻ đầy đủ thông qua FSDP
    * `--cpu-offload`: dỡ trạng thái trình tối ưu hóa và bản sao mô hình FP32 sang CPU (kết hợp với `--optimizer=cpu_adam`)
    * `--no-reshard-after-forward`: tăng tốc độ huấn luyện cho các mô hình lớn (1 tỷ+ tham số) và tương tự như ZeRO giai đoạn 2
    * các tùy chọn phổ biến khác (`--fp16`, `--update-freq`, `--checkpoint-activations`, `--offload-activations`, v.v.) tiếp tục hoạt động bình thường

**Hạn chế**

FSDP hiện có một số hạn chế so với backend DDP mặc định của fairseq (PyTorch DDP):

* trong khi FSDP hoàn toàn tương thích với các Trình tối ưu hóa theo điểm (pointwise Optimizers) (ví dụ: Adam, AdamW, Adadelta, Adamax, SGD, v.v.), hiện tại nó không tương thích với các Trình tối ưu hóa không theo điểm (ví dụ: Adagrad, Adafactor, LAMB, v.v.)
* FSDP phụ thuộc vào việc làm phẳng các tham số, vì vậy các mô hình hiện yêu cầu `--fp16-no-flatten-grads` có thể không được hỗ trợ


**Giải thích:**

FSDP là một kỹ thuật mới giúp cải thiện hiệu quả của huấn luyện mô hình AI quy mô lớn. Bằng cách chia sẻ các tham số mô hình và trạng thái trình tối ưu hóa trên nhiều GPU, FSDP giúp giảm thiểu việc sử dụng bộ nhớ và tăng tốc độ huấn luyện. 

FSDP đặc biệt hữu ích cho việc huấn luyện các mô hình cực lớn (với hàng tỷ tham số), vốn thường gặp khó khăn khi huấn luyện trên một số lượng GPU hạn chế.

**Keywords quan trọng:**

* **Fully Sharded Data Parallel (FSDP):** Chia sẻ dữ liệu song song hoàn toàn
* **Data parallel training:** Huấn luyện song song dữ liệu
* **Model parameters:** Tham số mô hình
* **Optimizer state:** Trạng thái trình tối ưu hóa
* **PyTorch DDP:**  Một kỹ thuật song song dữ liệu khác trong PyTorch
* **FP16/FP32:**  Các định dạng số dấu phẩy động


## gottbert

**GottBERT: a pure German language model**


## HuBERT

**Pre-trained and fine-tuned (ASR) models**

## joint_alignment_translation

**Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)**


Đây là tiêu đề một bài báo khoa học về dịch máy (Machine Translation) sử dụng mô hình Transformer.

**Giải thích:**

* **Jointly Learning:** Học cùng lúc hai nhiệm vụ, trong trường hợp này là căn chỉnh từ (word alignment) và dịch thuật (translation).
* **Word Alignment:**  Xác định sự tương ứng giữa các từ trong câu nguồn và câu đích. Ví dụ, trong câu "Tôi yêu bạn" (tiếng Việt) và "I love you" (tiếng Anh), "Tôi" tương ứng với "I", "yêu" tương ứng với "love", "bạn" tương ứng với "you".
* **Transformer Models:**  Kiến trúc mạng nơ-ron mạnh mẽ, thường được sử dụng trong các tác vụ xử lý ngôn ngữ tự nhiên, bao gồm dịch máy.

**Ý tưởng chính:**

Bài báo đề xuất một phương pháp huấn luyện mô hình Transformer để thực hiện đồng thời hai nhiệm vụ:

1. **Dịch thuật:**  Dịch câu từ ngôn ngữ nguồn sang ngôn ngữ đích.
2. **Căn chỉnh từ:**  Tìm ra sự tương ứng giữa các từ trong hai câu.

Việc học đồng thời hai nhiệm vụ này giúp mô hình hiểu rõ hơn về mối quan hệ giữa các từ và cấu trúc câu trong hai ngôn ngữ, từ đó cải thiện hiệu suất dịch thuật.

**Lợi ích:**

* **Cải thiện độ chính xác của bản dịch:**  Thông tin căn chỉnh từ giúp mô hình dịch chính xác hơn, đặc biệt là với các câu có cấu trúc phức tạp.
* **Hiểu rõ hơn về mối quan hệ giữa các ngôn ngữ:**  Việc căn chỉnh từ giúp phân tích sự tương đồng và khác biệt về cấu trúc ngữ pháp giữa các ngôn ngữ.

**Tóm lại:**

Bài báo "Jointly Learning to Align and Translate with Transformer Models" đã giới thiệu một phương pháp mới để huấn luyện mô hình dịch máy, kết hợp việc học căn chỉnh từ và dịch thuật, giúp cải thiện hiệu suất và hiểu biết về ngôn ngữ.

## language_model

**Neural Language Modeling**

## laser

**LASER Language-Agnostic SEntence Representations**

*LASER: Biểu diễn Câu Không phụ thuộc Ngôn ngữ (Language-Agnostic SEntence Representations)*

LASER là một thư viện để tính toán và sử dụng các nhúng câu đa ngôn ngữ (multilingual sentence embeddings).

Bạn có thể tìm thêm thông tin về LASER và cách sử dụng nó trên kho lưu trữ LASER chính thức.

Thư mục này chứa mã nguồn để huấn luyện nhúng LASER.

**Giải thích:**

* **Sentence Embeddings:**  Là kỹ thuật biểu diễn một câu văn dưới dạng một vector số thực. Các vector này mang thông tin ngữ nghĩa của câu, cho phép máy tính hiểu được ý nghĩa của câu.
* **Multilingual:**  Hỗ trợ nhiều ngôn ngữ. LASER có thể tạo ra các nhúng câu cho các câu văn ở nhiều ngôn ngữ khác nhau.
* **Language-Agnostic:**  Không phụ thuộc vào ngôn ngữ. Các nhúng câu được tạo ra bởi LASER có thể được so sánh trực tiếp với nhau, ngay cả khi chúng đến từ các ngôn ngữ khác nhau.

**Ứng dụng của LASER:**

* **Dịch máy (Machine Translation):**  So sánh các câu ở các ngôn ngữ khác nhau để tìm ra các câu tương ứng.
* **Truy vấn thông tin đa ngôn ngữ (Cross-lingual Information Retrieval):**  Tìm kiếm thông tin bằng một ngôn ngữ và nhận kết quả ở nhiều ngôn ngữ khác nhau.
* **Phân tích cảm xúc đa ngôn ngữ (Cross-lingual Sentiment Analysis):**  Phân tích cảm xúc của văn bản ở nhiều ngôn ngữ khác nhau.

**Tóm lại:**

LASER là một công cụ mạnh mẽ để xử lý ngôn ngữ tự nhiên đa ngôn ngữ. Nó cho phép tạo ra các biểu diễn câu không phụ thuộc vào ngôn ngữ, giúp máy tính hiểu và so sánh ý nghĩa của các câu văn ở nhiều ngôn ngữ khác nhau.

## latent_depth

**Deep Transformers with Latent Depth (Li et al., 2020)**

https://arxiv.org/abs/2009.13102.

*Transformer Sâu với Độ sâu Tiềm ẩn (Deep Transformers with Latent Depth)* (Li et al., 2020)

**Giới thiệu**

Chúng tôi trình bày một khung xác suất để tự động học lớp (các lớp) nào sẽ sử dụng bằng cách học phân phối hậu nghiệm (posterior distributions) của lựa chọn lớp. Mở rộng khung này, chúng tôi đề xuất một phương pháp mới để huấn luyện một mạng Transformer dùng chung cho dịch máy đa ngôn ngữ với các phân phối hậu nghiệm lựa chọn lớp khác nhau cho mỗi cặp ngôn ngữ.

**Giải thích:**

* **Deep Transformers:** Mô hình Transformer với nhiều lớp (layers). Mỗi lớp thực hiện một phép biến đổi trên dữ liệu đầu vào.
* **Latent Depth:**  Độ sâu tiềm ẩn, ý chỉ việc mô hình tự động học cách sử dụng số lượng lớp phù hợp cho từng tác vụ hoặc dữ liệu cụ thể.
* **Posterior Distributions:**  Phân phối hậu nghiệm, thể hiện xác suất lựa chọn mỗi lớp trong mô hình.

**Ý tưởng chính:**

Thay vì sử dụng tất cả các lớp trong mô hình Transformer, bài báo đề xuất một phương pháp để tự động học cách lựa chọn lớp phù hợp cho từng trường hợp. 

Ví dụ, khi dịch một câu đơn giản, mô hình có thể chỉ cần sử dụng một vài lớp đầu. Khi dịch một câu phức tạp, mô hình có thể cần sử dụng nhiều lớp hơn.

Việc học lựa chọn lớp được thực hiện thông qua việc học phân phối hậu nghiệm. Mỗi lớp sẽ có một xác suất được lựa chọn, xác suất này phụ thuộc vào dữ liệu đầu vào.

**Ứng dụng trong dịch máy đa ngôn ngữ:**

Phương pháp này cũng được áp dụng cho dịch máy đa ngôn ngữ. Mô hình sẽ học các phân phối hậu nghiệm lựa chọn lớp khác nhau cho mỗi cặp ngôn ngữ. Điều này giúp mô hình tối ưu hóa hiệu suất dịch cho từng cặp ngôn ngữ cụ thể.

**Lợi ích:**

* **Tăng hiệu quả:**  Chỉ sử dụng các lớp cần thiết, giúp giảm thiểu lượng tính toán và tăng tốc độ xử lý.
* **Cải thiện hiệu suất:**  Lựa chọn lớp phù hợp giúp mô hình hoạt động tốt hơn trên các tác vụ khác nhau.
* **Linh hoạt:**  Mô hình có thể tự thích ứng với các loại dữ liệu và tác vụ khác nhau.

**Tóm lại:**

Bài báo này giới thiệu một phương pháp mới để huấn luyện mô hình Transformer, cho phép mô hình tự động học cách sử dụng số lượng lớp phù hợp, từ đó tăng hiệu quả và hiệu suất trên các tác vụ xử lý ngôn ngữ tự nhiên.

## layerdrop

## linformer

## m2m_100

## mbart

## megatron_11b

## mms

## moe_lm

## mr_hubert

## multilingual

## noisychannel

## nonautoregressive_translation

## NormFormer

## operators

## paraphraser

## pay_less_attention_paper

## pointer_generator

## quant_noise

## roberta

## rfx

## scaling_nmt

## shuffled_word_order

## simultaneous_translation

## speech_recognition

## speech_synthesis

## speech_text_joint_to_text

## speech_to_speech

## speech_to_text

## stories

## textless_nlp

## translation

## translation_moe

## truncated_bptt

## unsupervised_quality_estimation

## wav2vec

## wmt19

## wmt20

## wmt21

## womens_bios

## xformers

## xglm

## xlmr

## xmod


