### Chương 1, Kiến trúc LLM

Trong chương này, bạn sẽ được giới thiệu về cấu trúc phức tạp của các mô hình ngôn ngữ lớn (LLM). Chúng tôi sẽ chia nhỏ kiến trúc LLM thành các phân đoạn dễ hiểu, tập trung vào các mô hình Transformer tiên tiến và các cơ chế chú ý quan trọng mà chúng sử dụng. Phân tích song song với các mô hình RNN trước đây sẽ cho phép bạn đánh giá cao sự phát triển và lợi thế của các kiến trúc hiện tại, đặt nền tảng cho sự hiểu biết kỹ thuật sâu hơn.

Trong chương này, chúng ta sẽ đề cập đến các chủ đề chính sau đây:

Cấu trúc của mô hình ngôn ngữ

Trình biến đổi và cơ chế chú ý

Mạng thần kinh hồi quy (RNN) và các hạn chế của chúng

Phân tích so sánh - Trình biến đổi so với mô hình RNN

Đến cuối chương này, bạn sẽ có thể hiểu được cấu trúc phức tạp của LLM, tập trung vào các mô hình Transformer tiên tiến và các cơ chế chú ý chính của chúng. Bạn cũng sẽ có thể nắm bắt được những cải tiến của kiến trúc hiện đại so với các mô hình RNN cũ, đây là bước tạo tiền đề cho sự hiểu biết kỹ thuật sâu sắc hơn về các hệ thống này.

### Cấu trúc của mô hình ngôn ngữ

Trong hành trình theo đuổi AI phản ánh chiều sâu và tính linh hoạt của giao tiếp con người, các mô hình ngôn ngữ như GPT-4 nổi lên như hình mẫu của ngôn ngữ học tính toán. Nền tảng của một mô hình như vậy là dữ liệu đào tạo của nó - một kho văn bản khổng lồ được rút ra từ văn học, phương tiện kỹ thuật số và vô số nguồn khác. Dữ liệu này không chỉ có số lượng lớn mà còn phong phú về chủng loại, bao gồm nhiều chủ đề, phong cách và ngôn ngữ để đảm bảo hiểu biết toàn diện về ngôn ngữ con người.

Cấu trúc của một mô hình ngôn ngữ như GPT-4 là minh chứng cho sự giao thoa giữa công nghệ phức tạp và sự tinh vi về ngôn ngữ. Mỗi thành phần, từ dữ liệu đào tạo đến tương tác với người dùng, phối hợp với nhau để tạo ra một mô hình không chỉ mô phỏng ngôn ngữ con người mà còn làm phong phú thêm cách chúng ta tương tác với máy móc. Chính thông qua cấu trúc phức tạp này mà các mô hình ngôn ngữ mang đến lời hứa về việc thu hẹp khoảng cách giao tiếp giữa con người và trí tuệ nhân tạo (AI).

Một mô hình ngôn ngữ như GPT-4 hoạt động trên một số lớp và thành phần phức tạp, mỗi lớp và thành phần phục vụ một chức năng duy nhất để hiểu, tạo và tinh chỉnh văn bản. Hãy cùng xem qua phần phân tích toàn diện.

### Dữ liệu đào tạo

Dữ liệu đào tạo cho một mô hình ngôn ngữ như GPT-4 là nền tảng để xây dựng khả năng hiểu và tạo ra ngôn ngữ con người của nó. Dữ liệu này được sắp xếp cẩn thận để bao gồm một loạt các kiến thức và biểu đạt của con người. Hãy thảo luận về các yếu tố chính cần xem xét khi đào tạo dữ liệu.

#### Phạm vi và sự đa dạng

Ví dụ: tập dữ liệu đào tạo cho GPT-4 bao gồm một kho văn bản khổng lồ được lựa chọn tỉ mỉ để bao quát càng nhiều phổ ngôn ngữ của con người càng tốt. Điều này bao gồm các khía cạnh sau:

* Các tác phẩm văn học: Tiểu thuyết, thơ ca, vở kịch và nhiều hình thức văn học tường thuật và phi tường thuật góp phần giúp mô hình hiểu các cấu trúc ngôn ngữ phức tạp, cách kể chuyện và cách sử dụng ngôn ngữ sáng tạo.
* Văn bản thông tin: Bách khoa toàn thư, tạp chí, báo cáo nghiên cứu và tài liệu giáo dục cung cấp cho mô hình kiến thức thực tế và kỹ thuật trong các lĩnh vực như khoa học, lịch sử, nghệ thuật và nhân văn.
* Nội dung web: Trang web cung cấp nhiều loại nội dung, bao gồm blog, bài báo, diễn đàn và nội dung do người dùng tạo. Điều này giúp mô hình học ngôn ngữ thông tục và tiếng lóng hiện tại, cũng như phương ngữ vùng miền và phong cách giao tiếp không chính thức.
* Nguồn đa ngôn ngữ: Để thành thạo nhiều ngôn ngữ, dữ liệu đào tạo bao gồm văn bản bằng nhiều ngôn ngữ khác nhau, góp phần vào khả năng dịch và hiểu văn bản không phải tiếng Anh của mô hình.
* Phương sai văn hóa: Văn bản từ các nền văn hóa và khu vực khác nhau làm phong phú thêm tập dữ liệu của mô hình với các sắc thái văn hóa và chuẩn mực xã hội.

#### Chất lượng và tuyển chọn

Chất lượng của dữ liệu đào tạo là rất quan trọng. Nó phải có các thuộc tính sau:

* Sạch sẽ: Dữ liệu phải không có lỗi, chẳng hạn như ngữ pháp không chính xác hoặc lỗi chính tả, trừ khi những lỗi này là cố ý và đại diện cho cách sử dụng ngôn ngữ nhất định.
* Chính xác: Độ chính xác là điều tối quan trọng. Dữ liệu phải chính xác và phản ánh thông tin trung thực để đảm bảo độ tin cậy của kết quả đầu ra của Al.
* Đa dạng: Việc bao gồm các phong cách viết đa dạng, từ giọng điệu trang trọng đến giọng điệu đàm thoại, đảm bảo rằng mô hình có thể điều chỉnh phản hồi của mình cho phù hợp với các ngữ cảnh khác nhau.
* Cân bằng: Không có thể loại hoặc nguồn nào được chi phối tập dữ liệu đào tạo để ngăn chặn sự thiên vị trong tạo ngôn ngữ.
* Có tính đại diện: Dữ liệu phải đại diện cho vô số cách sử dụng ngôn ngữ trong các lĩnh vực và nhân khẩu học khác nhau để tránh hiểu sai về các mẫu ngôn ngữ.

#### Quá trình đào tạo

Việc đào tạo thực tế bao gồm việc đưa dữ liệu văn bản vào mô hình, sau đó mô hình sẽ học cách dự đoán từ tiếp theo trong một chuỗi dựa trên các từ đứng trước nó. Quá trình này, được gọi là học có giám sát, không yêu cầu dữ liệu được gắn nhãn mà thay vào đó dựa vào các mẫu có sẵn trong chính văn bản.

#### Những thách thức và giải pháp

Những thách thức và giải pháp liên quan đến quá trình đào tạo như sau:

* Thiên vị: Các mô hình ngôn ngữ có thể vô tình học và duy trì các thành kiến hiện có trong dữ liệu đào tạo. Để khắc phục điều này, các tập dữ liệu thường được kiểm tra về độ chệch và những nỗ lực được thực hiện để đưa vào một đại diện cân bằng.
* Thông tin sai lệch: Văn bản chứa thông tin không chính xác về mặt thực tế có thể khiến mô hình học thông tin không chính xác. Các nhà tuyển chọn dữ liệu đặt mục tiêu bao gồm các nguồn đáng tin cậy và có thể sử dụng các kỹ thuật lọc để giảm thiểu việc đưa vào thông tin sai lệch.
* Cập nhật kiến thức: Khi ngôn ngữ phát triển và thông tin mới xuất hiện, tập dữ liệu đào tạo phải được cập nhật. Điều này có thể bao gồm việc thêm các văn bản gần đây hoặc sử dụng các kỹ thuật để cho phép mô hình liên tục học hỏi từ dữ liệu mới.

Dữ liệu đào tạo cho GPT-4 là nền tảng củng cố khả năng ngôn ngữ của nó. Nó phản ánh kiến thức của con người và sự đa dạng về ngôn ngữ, cho phép mô hình thực hiện nhiều tác vụ liên quan đến ngôn ngữ với sự lưu loát đáng kể. Quá trình xử lý, cân bằng và cập nhật dữ liệu đang diễn ra này cũng quan trọng như việc phát triển kiến trúc của chính mô hình, đảm bảo rằng mô hình ngôn ngữ vẫn là một công cụ năng động và chính xác để hiểu và tạo ra ngôn ngữ con người.

### Tokenization

Mã hóa là một bước tiền xử lý cơ bản trong quá trình đào tạo các mô hình ngôn ngữ như GPT-4, đóng vai trò như cầu nối giữa văn bản thô và các thuật toán số làm nền tảng cho học máy (ML). Mã hóa là một bước tiền xử lý quan trọng trong việc đào tạo các mô hình ngôn ngữ. Nó ảnh hưởng đến khả năng hiểu văn bản của mô hình và ảnh hưởng đến hiệu suất tổng thể của các tác vụ liên quan đến ngôn ngữ. Khi các mô hình như GPT-4 được đào tạo trên các tập dữ liệu ngày càng đa dạng và phức tạp, các chiến lược mã hóa tiếp tục phát triển, nhằm mục đích tối đa hóa hiệu quả và độ chính xác trong việc thể hiện ngôn ngữ của con người.

Dưới đây là một số thông tin chuyên sâu về mã hóa:

* Hiểu về mã hóa: Mã hóa là quá trình chuyển đổi một chuỗi ký tự thành một chuỗi mã thông báo, có thể được coi là các khối xây dựng của văn bản. Mã thông báo là một chuỗi các ký tự liền nhau, được phân cách bởi dấu cách hoặc dấu chấm câu, được coi là một nhóm. Trong mô hình hóa ngôn ngữ, mã thông báo thường là các từ, nhưng chúng cũng có thể là các phần của từ (chẳng hạn như từ phụ hoặc hình vị), dấu chấm câu hoặc thậm chí toàn bộ câu.
* Vai trò của mã thông báo: Mã thông báo là các đơn vị nhỏ nhất mang ý nghĩa trong văn bản. Theo thuật ngữ tính toán, chúng là các yếu tố nguyên tử mà mô hình ngôn ngữ sử dụng để hiểu và tạo ra ngôn ngữ. Mỗi mã thông báo được liên kết với một vectơ trong mô hình, nắm bắt thông tin ngữ nghĩa và cú pháp về mã thông báo đó trong một không gian nhiều chiều.
* Mã hóa:
    * Mã hóa cấp từ: Đây là hình thức đơn giản nhất, trong đó văn bản được chia thành các mã thông báo dựa trên dấu cách và dấu chấm câu. Mỗi từ trở thành một mã thông báo.
    * Mã hóa từ phụ: Để giải quyết các thách thức của mã hóa cấp từ, chẳng hạn như xử lý các từ không xác định, các mô hình ngôn ngữ thường sử dụng mã hóa từ phụ. Điều này bao gồm việc chia nhỏ các từ thành các đơn vị nhỏ hơn có ý nghĩa (từ phụ), giúp mô hình khái quát hóa tốt hơn cho các từ mới. Điều này đặc biệt hữu ích để xử lý các ngôn ngữ biến tố, trong đó cùng một từ gốc có thể có nhiều biến thể.
    * Mã hóa cặp byte (BPE): BPE là một phương pháp mã hóa từ phụ phổ biến. Nó bắt đầu với một kho văn bản lớn và kết hợp các cặp ký tự xuất hiện thường xuyên nhất một cách lặp đi lặp lại. Điều này tiếp tục cho đến khi một tập hợp các đơn vị từ phụ được xây dựng để tối ưu hóa cho các mẫu phổ biến nhất của kho văn bản.
    * SentencePiece: SentencePiece là một thuật toán mã hóa không dựa vào ranh giới từ được xác định trước và có thể hoạt động trực tiếp trên văn bản thô. Điều này có nghĩa là nó xử lý văn bản ở dạng thô mà không cần phân đoạn trước thành các từ. Phương pháp này khác với các phương pháp như BPE, thường yêu cầu phân đoạn văn bản ban đầu. Hoạt động trực tiếp trên văn bản thô cho phép SentencePiece không phụ thuộc vào ngôn ngữ, làm cho nó đặc biệt hiệu quả đối với các ngôn ngữ không sử dụng khoảng trắng để phân tách các từ, chẳng hạn như tiếng Nhật hoặc tiếng Trung. Ngược lại, BPE thường hoạt động trên văn bản được mã hóa trước, trong đó các từ đã được phân tách, điều này có thể hạn chế hiệu quả của nó đối với một số ngôn ngữ nhất định mà không có ranh giới từ rõ ràng. Bằng cách không phụ thuộc vào ranh giới được xác định trước, SentencePiece có thể xử lý nhiều loại ngôn ngữ và tập lệnh khác nhau, cung cấp phương pháp mã hóa linh hoạt và mạnh mẽ hơn cho các ngữ cảnh ngôn ngữ đa dạng.

* Quá trình mã hóa

Quá trình mã hóa trong ngữ cảnh của các mô hình ngôn ngữ bao gồm một số bước:

1. Phân đoạn: Chia văn bản thành các mã thông báo dựa trên các quy tắc được xác định trước hoặc các mẫu đã học.
2. Chuẩn hóa: Đôi khi, mã thông báo được chuẩn hóa thành một dạng tiêu chuẩn. Ví dụ: 'USA' và 'U.S.A.' có thể được chuẩn hóa thành một hình thức duy nhất.
3. Lập chỉ mục từ vựng: Mỗi mã thông báo duy nhất được liên kết với một chỉ mục trong danh sách từ vựng. Mô hình sẽ sử dụng các chỉ mục này, không phải bản thân văn bản, để xử lý ngôn ngữ.
4. Biểu diễn vectơ: Mã thông báo được chuyển đổi thành biểu diễn số, thường là vectơ một nóng hoặc vectơ nhúng, sau đó được đưa vào mô hình.

* Tầm quan trọng của mã hóa

Mã hóa đóng một vai trò quan trọng trong hiệu suất của các mô hình ngôn ngữ bằng cách hỗ trợ các khía cạnh sau:

* Hiệu quả: Nó cho phép mô hình xử lý lượng lớn văn bản một cách hiệu quả bằng cách giảm kích thước của từ vựng mà nó cần xử lý.
* Xử lý các từ không xác định: Bằng cách chia nhỏ các từ thành các đơn vị từ phụ, mô hình có thể xử lý các từ mà nó chưa từng thấy trước đây, điều này đặc biệt quan trọng đối với các mô hình miền mở gặp phải văn bản đa dạng.
* Tính linh hoạt của ngôn ngữ: Mã hóa từ phụ và cấp ký tự cho phép mô hình hoạt động với nhiều ngôn ngữ hiệu quả hơn mã hóa cấp từ. Điều này là do các phương pháp từ phụ và cấp ký tự chia nhỏ văn bản thành các đơn vị nhỏ hơn, có thể nắm bắt các điểm chung giữa các ngôn ngữ và xử lý các tập lệnh và cấu trúc khác nhau. Ví dụ: nhiều ngôn ngữ có chung từ gốc, tiền tố và hậu tố có thể hiểu được ở cấp độ từ phụ. Mức độ chi tiết này giúp mô hình khái quát hóa tốt hơn giữa các ngôn ngữ, bao gồm cả những ngôn ngữ có hình thái phong phú hoặc tập lệnh độc đáo.
* Học ngữ nghĩa và cú pháp: Mã hóa thích hợp cho phép mô hình tìm hiểu mối quan hệ giữa các mã thông báo khác nhau, nắm bắt các sắc thái của ngôn ngữ.

* Những thách thức của mã hóa

Các thách thức sau đây liên quan đến mã hóa:

* Mơ hồ: Mã hóa có thể mơ hồ, đặc biệt là trong các ngôn ngữ có cấu tạo từ phức tạp hoặc trong trường hợp từ đồng âm (các từ được đánh vần giống nhau nhưng có nghĩa khác nhau)
* Phụ thuộc vào ngữ cảnh: Ý nghĩa của mã thông báo có thể phụ thuộc vào ngữ cảnh của nó, điều này không phải lúc nào cũng được xem xét trong các lược đồ mã hóa đơn giản
* Sự khác biệt về văn hóa: Các nền văn hóa khác nhau có thể có các nhu cầu mã hóa khác nhau, chẳng hạn như từ ghép trong tiếng Đức hoặc thiếu dấu cách trong tiếng Trung

### Kiến trúc mạng thần kinh

Kiến trúc mạng thần kinh của các mô hình như GPT-4 là một hệ thống phức tạp và tinh vi được thiết kế để xử lý và tạo ra ngôn ngữ con người với trình độ thành thạo cao. Kiến trúc thần kinh Transformer, là xương sống của GPT-4, đại diện cho một bước nhảy vọt đáng kể trong quá trình phát triển của các thiết kế mạng thần kinh để xử lý ngôn ngữ.

* Kiến trúc Transformer

Kiến trúc Transformer được giới thiệu trong một bài báo có tiêu đề Attention Is All You Need, bởi Vaswani et al., vào năm 2017. Nó đại diện cho sự khác biệt so với các mô hình chuỗi-chuỗi trước đó sử dụng mạng thần kinh hồi quy (RNN) hoặc mạng thần kinh tích chập (CNN). Mô hình Transformer được thiết kế để xử lý dữ liệu tuần tự mà không cần các cấu trúc hồi quy này, do đó cho phép song song hóa nhiều hơn và giảm đáng kể thời gian đào tạo. Transformer hoàn toàn dựa vào cơ chế tự chú ý để xử lý dữ liệu song song, cho phép tính toán nhanh hơn đáng kể.

* Cơ chế tự chú ý

Bộ mã hóa xử lý dữ liệu đầu vào thành một biểu diễn cố định để mô hình sử dụng thêm, trong khi bộ giải mã biến đổi biểu diễn cố định trở lại thành định dạng đầu ra mong muốn, chẳng hạn như văn bản hoặc chuỗi. Tự chú ý, đôi khi được gọi là chú ý nội bộ, là một cơ chế cho phép mỗi vị trí trong bộ mã hóa tham dự tất cả các vị trí trong lớp trước đó của bộ mã hóa. Tương tự, mỗi vị trí trong bộ giải mã có thể tham dự tất cả các vị trí trong bộ mã hóa và tất cả các vị trí cho đến và bao gồm cả vị trí đó trong bộ giải mã. Cơ chế này rất quan trọng đối với khả năng hiểu ngữ cảnh và mối quan hệ trong dữ liệu đầu vào của mô hình.

* Tự chú ý tại nơi làm việc

Nó tính toán một tập hợp các điểm chú ý cho mỗi mã thông báo trong dữ liệu đầu vào, xác định mức độ tập trung mà nó nên đặt vào các phần khác của đầu vào khi xử lý một mã thông báo cụ thể. Những điểm số này được sử dụng để tạo ra một tổ hợp có trọng số của các vectơ giá trị, sau đó trở thành đầu vào cho lớp tiếp theo hoặc đầu ra của mô hình.

* Đa đầu tự chú ý

Một khía cạnh quan trọng của cơ chế chú ý của Transformer là nó sử dụng nhiều "đầu", có nghĩa là nó chạy cơ chế chú ý nhiều lần song song. Mỗi "đầu" học các khía cạnh khác nhau của dữ liệu, cho phép mô hình nắm bắt các loại phụ thuộc khác nhau trong đầu vào: cú pháp, ngữ nghĩa và vị trí. Ưu điểm của sự chú ý đa đầu như sau:

* Nó cung cấp cho mô hình khả năng chú ý đến các phần khác nhau của chuỗi đầu vào một cách khác nhau, tương tự như việc xem xét một vấn đề từ các quan điểm khác nhau
* Nhiều biểu diễn của mỗi mã thông báo được học, làm phong phú thêm sự hiểu biết của mô hình về mỗi mã thông báo trong ngữ cảnh của nó

* Mạng nạp chuyển tiếp theo vị trí

Sau các lớp con chú ý trong mỗi lớp của bộ mã hóa và bộ giải mã, có một mạng nạp chuyển tiếp được kết nối đầy đủ. Mạng này áp dụng cùng một phép biến đổi tuyến tính cho từng vị trí riêng biệt và giống hệt nhau. Phần này của mô hình có thể được coi là một bước xử lý giúp tinh chỉnh đầu ra của cơ chế chú ý trước khi chuyển nó sang lớp tiếp theo.

Chức năng của mạng nạp chuyển tiếp là cung cấp cho mô hình khả năng áp dụng các phép biến đổi phức tạp hơn cho dữ liệu. Phần này của mô hình có thể học và biểu diễn các phụ thuộc phi tuyến tính trong dữ liệu, điều này rất quan trọng để nắm bắt sự phức tạp của ngôn ngữ.

* Chuẩn hóa lớp và kết nối còn lại

Kiến trúc Transformer sử dụng chuẩn hóa lớp và kết nối còn lại để tăng cường tính ổn định của quá trình đào tạo và cho phép đào tạo các mô hình sâu hơn:

* Chuẩn hóa lớp: Nó chuẩn hóa các đầu vào trên các tính năng cho mỗi mã thông báo một cách độc lập và được áp dụng trước mỗi lớp con trong Transformer, tăng cường tính ổn định của quá trình đào tạo và hiệu suất của mô hình.
* Kết nối còn lại: Mỗi lớp con trong Transformer, có thể là cơ chế chú ý hoặc mạng nạp chuyển tiếp, đều có kết nối còn lại xung quanh nó, sau đó là chuẩn hóa lớp. Điều này có nghĩa là đầu ra của mỗi lớp con được thêm vào đầu vào của nó trước khi được chuyển tiếp, điều này giúp giảm thiểu vấn đề gradient biến mất, cho phép các kiến trúc sâu hơn. Vấn đề gradient biến mất xảy ra trong quá trình đào tạo mạng thần kinh sâu khi gradient của hàm mất mát giảm theo cấp số nhân khi chúng được lan truyền ngược qua các lớp, dẫn đến các bản cập nhật trọng số cực kỳ nhỏ và cản trở việc học.

Kiến trúc mạng thần kinh của GPT-4, dựa trên Transformer, là minh chứng cho sự phát triển của các kỹ thuật ML trong xử lý ngôn ngữ tự nhiên (NLP). Cơ chế tự chú ý cho phép mô hình tập trung vào các phần khác nhau của đầu vào, sự chú ý đa đầu cho phép nó nắm bắt nhiều loại phụ thuộc và mạng nạp chuyển tiếp theo vị trí góp phần hiểu các mẫu phức tạp. Chuẩn hóa lớp và kết nối còn lại đảm bảo rằng mô hình có thể được đào tạo hiệu quả ngay cả khi nó rất sâu. Tất cả các thành phần này phối hợp với nhau hài hòa để cho phép các mô hình như GPT-4 tạo ra văn bản phong phú về ngữ cảnh, mạch lạc và thường không thể phân biệt được với văn bản do con người viết.

### Nhúng

Trong ngữ cảnh của các mô hình ngôn ngữ như GPT-4, nhúng là một thành phần quan trọng cho phép các mô hình này xử lý và hiểu văn bản ở cấp độ toán học. Nhúng biến đổi các mã thông báo rời rạc như từ, từ phụ hoặc ký tự thành các vectơ liên tục, từ đó có thể áp dụng phép toán vectơ cho các vectơ nhúng. Hãy chia nhỏ khái niệm nhúng và vai trò của chúng trong các mô hình ngôn ngữ:

* Nhúng từ: Nhúng từ là dạng nhúng trực tiếp nhất, trong đó mỗi từ trong từ vựng của mô hình được chuyển đổi thành một vectơ nhiều chiều. Các vectơ này được học trong quá trình đào tạo.

Hãy xem xét các đặc điểm của nhúng từ:

* Biểu diễn dày đặc: Mỗi từ được biểu diễn bằng một vectơ dày đặc, thường có vài trăm chiều, trái ngược với các biểu diễn thưa thớt, nhiều chiều như mã hóa một nóng.
* Độ tương tự ngữ nghĩa: Các từ tương tự về mặt ngữ nghĩa có xu hướng có các vectơ nhúng gần nhau trong không gian vectơ. Điều này cho phép mô hình hiểu các từ đồng nghĩa, phép loại suy và các mối quan hệ ngữ nghĩa chung.
* Được học theo ngữ cảnh: Các vectơ nhúng được học dựa trên ngữ cảnh mà các từ xuất hiện, do đó, vectơ của một từ không chỉ nắm bắt bản thân từ đó mà còn nắm bắt cách sử dụng từ đó.
* Nhúng từ phụ: Để xử lý các từ ngoài từ vựng và các ngôn ngữ giàu hình thái, nhúng từ phụ sẽ chia nhỏ các từ thành các thành phần nhỏ hơn. Điều này cho phép mô hình tạo ra các vectơ nhúng cho các từ mà nó chưa từng thấy trước đây, dựa trên các đơn vị từ phụ.
* Nhúng vị trí: Vì kiến trúc Transformer mà GPT-4 sử dụng không xử lý dữ liệu tuần tự theo thứ tự, nên nhúng vị trí được thêm vào để cung cấp cho mô hình thông tin về vị trí của các từ trong một chuỗi.

Hãy xem xét các đặc điểm của nhúng vị trí:

* Thông tin tuần tự: Nhúng vị trí mã hóa thứ tự của các mã thông báo trong chuỗi, cho phép mô hình phân biệt giữa "John chơi piano" và "Piano chơi John", chẳng hạn.
* Được thêm vào nhúng từ: Các vectơ vị trí này thường được thêm vào nhúng từ trước khi chúng được đưa vào các lớp Transformer, đảm bảo rằng thông tin vị trí được chuyển qua mô hình.

Để hiểu kiến trúc của các mô hình ngôn ngữ, chúng ta phải hiểu hai thành phần cơ bản:

* Lớp đầu vào: Trong các mô hình ngôn ngữ, nhúng tạo thành lớp đầu vào, biến đổi mã thông báo thành một định dạng mà mạng thần kinh có thể làm việc
* Quá trình đào tạo: Trong quá trình đào tạo, các vectơ nhúng được điều chỉnh cùng với các tham số khác của mô hình để giảm thiểu hàm mất mát, do đó tinh chỉnh khả năng nắm bắt thông tin ngôn ngữ của chúng

Sau đây là hai giai đoạn quan trọng trong quá trình phát triển và cải tiến các mô hình ngôn ngữ:

* Khởi tạo: Các vectơ nhúng có thể được khởi tạo ngẫu nhiên và học từ đầu trong quá trình đào tạo hoặc chúng có thể được đào tạo trước bằng cách sử dụng phương pháp học không giám sát trên một kho văn bản lớn và sau đó được tinh chỉnh cho các tác vụ cụ thể.
* Học chuyển giao: Các vectơ nhúng có thể được chuyển giao giữa các mô hình hoặc tác vụ khác nhau. Đây là nguyên tắc đằng sau các mô hình như BERT, trong đó các vectơ nhúng được học từ một tác vụ có thể được áp dụng cho một tác vụ khác.

#### Những thách thức và giải pháp

Có những thách thức bạn phải vượt qua khi sử dụng nhúng. Hãy xem qua chúng và tìm hiểu cách giải quyết chúng:

* Tính chiều cao: Các vectơ nhúng có chiều cao, điều này có thể khiến chúng tốn kém về mặt tính toán. Có thể sử dụng các kỹ thuật giảm chiều và phương pháp đào tạo hiệu quả để quản lý điều này.
* Phụ thuộc vào ngữ cảnh: Một từ có thể có nghĩa khác nhau trong các ngữ cảnh khác nhau. Các mô hình như GPT-4 sử dụng ngữ cảnh xung quanh để điều chỉnh các vectơ nhúng trong giai đoạn tự chú ý, giải quyết thách thức này.

Tóm lại, nhúng là một yếu tố nền tảng của các mô hình ngôn ngữ hiện đại, biến đổi nguyên liệu thô của văn bản thành một dạng toán học phong phú, sắc thái mà mô hình có thể học hỏi. Bằng cách nắm bắt ý nghĩa ngữ nghĩa và mã hóa thông tin vị trí, nhúng cho phép các mô hình như GPT-4 tạo và hiểu ngôn ngữ với mức độ tinh vi đáng kể.