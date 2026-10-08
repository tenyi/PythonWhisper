import pytest
import requests
from unittest.mock import patch, Mock
from translator_ollama import summary_text_ollama, correct_words_ollama


class TestSummaryTextOllama:
    """測試 summary_text_ollama 函數"""

    def test_summary_text_ollama_long_text(self):
        """測試長文字總結"""
        input_text = " Hello everyone. Thanks for joining. My name is Titus Kurek. I am a product manager at Canonical and today together with our partner from cloud-based solutions, Gabriel Samfira, we are going to present a webinar from VMware to Charmed OpenStack. We are going to navigate through a comparison between VMware and OpenStack suites and try to find a common ground so that we could compare apples to apples. Later, I am going to show the benefits of using Charmed OpenStack such as efficiency and flexibility and discuss what makes Charmed OpenStack different from any other OpenStacks available there. The story begins a few years ago when organizations have started virtualizing their IT workloads and migrating from legacy monolithic infrastructure to cloud environments. The migration was mostly driven by a constantly growing TCO associated with maintaining the legacy infrastructure. In search of reduced costs many chose VMware as a provider for their virtualized infrastructure. They were hoping that moving to virtualized environments and running workloads on a shared hardware is sufficient to get both CAPEX and OPEX costs down. However, because of the costs associated with VMware licensing, support and professional services many are not able to meet their primary goal significantly reduced TCO. Instead, they found out that the TCO is still growing. In search of alternative solutions, organizations have recently started exploring other platforms such as OpenStack. So let's now have a look on these two technologies that matter today. We're going to start with the VMware Virtualization Platform. If you research VMware, you're going to find out that what VMware offers today is so-called vRealize Suite, which is available in three different types. What is common between these three is a licensing model which currently is based on a portable license unit or a PLU, which is assigned for either one vSphere CPU or 15 non-vSphere OSs, which basically means that for every 15 virtual machines or for every CPU, you need a separate portable license unit. For the purposes of today we are going to focus on the advanced offering of VMware vRealize Suite or the advanced version and this is because it is most closely resembling what OpenStack provides today and even more importantly what Charmed OpenStack provides today so that we would be able to make the apples to apples comparison. When we take a look on the VMware suite there are various components which build it up. What we have on the top are components responsible for hypervisor and SDN management and vRealize orchestration plus VMware NSX and SDN itself. While what follows in the middle are all components responsible for performance and capacity management. There is an IAAS component. There is an IAAS component, vRealize automation and various components which build the business continuity and disaster recovery piece. vSphere replication, vSphere data protection and site recovery manager. In turn when we take a look on the Charmed OpenStack suite it consists of the following components. Let's start from the bottom. So you can run Charmed OpenStack on top of certified hardware or you can run it in a public cloud or you can obviously also run it on top of your VMware virtualization environment. We use Ubuntu operating system as a base and now as depicted on the right there is Charmed OpenStack Infrastructure as a Service Cloud. It consists of the following components compute, network, storage and other OpenStack shared services. But you can also use other hypervisors. As for the SDN it's really an SDN of your choice and for storage purposes you can use Ceph and Swift for image, block and object storage respectively. Now on top of the OpenStack cloud you can run your application workloads but you can also run them on top of Kubernetes. So Kubernetes is usually deployed on top of OpenStack but you can also run Kubernetes directly on bare metal. And now as depicted on the left there is a bunch of Ubuntu cloud tools which are used to operate the whole environment. Those consist of three main components service management, log management and monitoring. As for the service management we use mass metal as a service which is a server provisioning tool which basically allows you to automate bare metal provisioning. And we use Juju the application modeling tool which allows you to deploy and operate your OpenStack cloud in an efficient and flexible way. As for the log management we recommend the ELK stack Elasticsearch, Logstash and Kibana while for monitoring you can use Landscape, Nagios and Prometheus. Now as we know the components which build up VMware vRealize suite and Charmed OpenStack suite we can make a detailed comparison between these two platforms and we can do it by the features they provide. So let's start with the hypervisor management and self-service portal. OpenStack has two building components Nova and Horizon which provide these two features while on the VMware side we have vCenter server and vRealize automation. For orchestration and application deployments Canonical provides Juju the application modeling tool which can be used to model and deploy and orchestrate workloads on both infrastructure and application layer. For monitoring and block management purposes we provide the LMA stack that was presented on a previous slide. Charmed OpenStack does not have any built-in billing tool but you can easily integrate any third-party tools with it. And now what makes the core difference between VMware and Charmed OpenStack and basically any other OpenStack available on the market are the hypervisor and SDN. VMware provides hypervisor and SDN that it brings those are ESXi and NSX while in Charmed OpenStack we can use KVM by default but you can also use any other hypervisor that is available and the same story applies to SDN. You can basically use any SDN that is currently available on the market. So at this point one can ask a question what about VMware integrated OpenStack. VMware provides an integrated OpenStack that you can run on top of its virtualization infrastructure. There are some drawbacks associated with this solution however which includes vendor locking. You are still running on top of VMware aren't you? It adds more complexity to complex enough environment. You do not benefit from reduced costs. You still have to pay for VMware licensing, support and subscription and professional services. And finally you do not benefit from other features of Charmed OpenStack. So let's move to the next section now and talk more about what makes Charmed OpenStack different. Among various reasons to choose Charmed OpenStack these top 5 are the most important to remember. Number one technology choices of Charmed OpenStack are very wide especially in terms of networking, storage and hypervisor but also including integration with disaster recovery and backup solutions and migration tools. Number two predictable release cadence So Charmed OpenStack has very predictable release cadence. Basically every six months there's a new version of OpenStack available and within the weeks of release Canonical provides support for the new version. Moreover every two years there's an LTS long time supported version of OpenStack released by Canonical. And finally OpenStack releases are tied into Ubuntu releases so there is a clear upgrade path from the older versions to the newer ones. There is a high velocity of development and innovation in Charmed OpenStack as it usually is in any other open source projects. Apart from the core OpenStack components such as Nova, Neutron, Keystone, Swift, Cinder and Glanz Charmed OpenStack also provides full support for the components like Horizon Dashboard, Manila shared file system, Barbican key management, Sailometer for telemetry, heat war, workloads orchestration and designate DNS service. One of the core reasons to choose Charmed OpenStack are simplified deployments and operations thanks to the usage of juju charms. A charm is a collection of scripts and metadata which contain all necessary logic required to install, configure and operate your applications. They basically contain a distilled knowledge of experts in the field from Canonical and other companies which is written in the form of a code. Charmed also uses so-called declarative DevOps approach which contrary to the imperative DevOps approach allows you to focus on the what instead of the how question. By using declarative DevOps approach you can easily deploy your OpenStack by just specifying a number of machines to be deployed, placing services across the machines, adding or removing certain services or turning on or off certain features. This model is later defined in the form of a yaml file which is later used by juju to deploy your OpenStack. While certain operations are performed by charms, the juju controller is managing all of those operations. Charms can also significantly simplify daily operational tasks such as OpenStack upgrades, database backups or scaling out your cloud environment. Last but not least, economics as always plays a huge role. By using Charmed OpenStack you can ensure reduction of both capex and opex costs."
        expected_summary = "討論新產品開發計畫，下月開始執行，負責人張三。"
        result = summary_text_ollama(input_text, "http://10.10.10.201:11434/api/chat")
        assert result == expected_summary

    @patch("translator_ollama.requests.post")
    @patch("translator_ollama.OpenCC")
    def test_summary_text_ollama_success(self, mock_opencc, mock_post):
        """測試成功總結會議摘要"""
        # 準備測試資料
        input_text = (
            "今天的會議討論了新產品的開發計畫，決定在下個月開始執行，負責人是張三。"
        )
        expected_summary = "討論新產品開發計畫，下月開始執行，負責人張三。"

        # 模擬 API 回應
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": expected_summary}}
        mock_post.return_value = mock_response

        # 模擬 OpenCC 轉換
        mock_cc_instance = Mock()
        mock_cc_instance.convert.return_value = expected_summary
        mock_opencc.return_value = mock_cc_instance

        # 執行測試
        result = summary_text_ollama(input_text, "http://10.10.10.201:11434/api/chat")

        # 驗證結果
        assert result == expected_summary

        # 驗證 API 呼叫
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "gemma3:27b"
        assert call_args[1]["json"]["messages"][1]["content"] == input_text
        assert call_args[1]["json"]["stream"] is False

        # 驗證 OpenCC 呼叫
        mock_opencc.assert_called_once_with("s2twp")
        mock_cc_instance.convert.assert_called_once_with(expected_summary)

    @patch("translator_ollama.requests.post")
    @patch("translator_ollama.OpenCC")
    def test_summary_text_ollama_with_custom_params(self, mock_opencc, mock_post):
        """測試使用自定義參數的會議摘要"""
        # 準備測試資料
        input_text = "會議內容"
        custom_url = "http://custom-server:11434/api/chat"
        custom_model = "llama2:7b"
        expected_summary = "自定義摘要結果"

        # 模擬 API 回應
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": expected_summary}}
        mock_post.return_value = mock_response

        # 模擬 OpenCC 轉換
        mock_cc_instance = Mock()
        mock_cc_instance.convert.return_value = expected_summary
        mock_opencc.return_value = mock_cc_instance

        # 執行測試
        result = summary_text_ollama(input_text, url=custom_url, model=custom_model)

        # 驗證結果
        assert result == expected_summary

        # 驗證 API 呼叫使用了自定義參數
        call_args = mock_post.call_args
        assert call_args[0][0] == custom_url
        assert call_args[1]["json"]["model"] == custom_model

    @patch("translator_ollama.requests.post")
    def test_summary_text_ollama_api_error(self, mock_post):
        """測試 API 錯誤處理"""
        # 準備測試資料
        input_text = "測試文字"

        # 模擬 API 錯誤
        mock_post.side_effect = requests.exceptions.RequestException("API 連線錯誤")

        # 執行測試並驗證例外
        with pytest.raises(requests.exceptions.RequestException):
            summary_text_ollama(input_text)

    @patch("translator_ollama.requests.post")
    @patch("translator_ollama.OpenCC")
    def test_summary_text_ollama_empty_response(self, mock_opencc, mock_post):
        """測試空回應處理"""
        # 準備測試資料
        input_text = "測試文字"

        # 模擬空回應
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "content": "   "  # 只有空白字符
            }
        }
        mock_post.return_value = mock_response

        # 模擬 OpenCC 轉換
        mock_cc_instance = Mock()
        mock_cc_instance.convert.return_value = ""
        mock_opencc.return_value = mock_cc_instance

        # 執行測試
        result = summary_text_ollama(input_text)

        # 驗證結果為空字串
        assert result == ""

    @patch("translator_ollama.requests.post")
    @patch("translator_ollama.OpenCC")
    def test_summary_text_ollama_system_prompt_content(self, mock_opencc, mock_post):
        """測試系統提示詞內容是否正確"""
        # 準備測試資料
        input_text = "測試會議內容"
        expected_summary = "測試摘要"

        # 模擬 API 回應
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": expected_summary}}
        mock_post.return_value = mock_response

        # 模擬 OpenCC 轉換
        mock_cc_instance = Mock()
        mock_cc_instance.convert.return_value = expected_summary
        mock_opencc.return_value = mock_cc_instance

        # 執行測試
        summary_text_ollama(input_text)

        # 驗證系統提示詞包含關鍵內容
        call_args = mock_post.call_args
        system_message = call_args[1]["json"]["messages"][0]["content"]

        # 檢查系統提示詞包含重要指令
        assert (
            "總結會議摘要" in system_message or "summarizing meetings" in system_message
        )
        assert "Traditional Chinese" in system_message or "繁體中文" in system_message
        assert "無法得出會議摘要" in system_message
        assert "zh-TW" in system_message

    @patch("translator_ollama.requests.post")
    @patch("translator_ollama.OpenCC")
    def test_summary_text_ollama_opencc_conversion(self, mock_opencc, mock_post):
        """測試 OpenCC 簡繁轉換功能"""
        # 準備測試資料
        input_text = "会议内容"
        api_response = "这是简体中文的摘要"
        expected_traditional = "這是繁體中文的摘要"

        # 模擬 API 回應
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": api_response}}
        mock_post.return_value = mock_response

        # 模擬 OpenCC 轉換
        mock_cc_instance = Mock()
        mock_cc_instance.convert.return_value = expected_traditional
        mock_opencc.return_value = mock_cc_instance

        # 執行測試
        result = summary_text_ollama(input_text, "http://10.10.10.201:11434/api/chat")

        # 驗證 OpenCC 被正確呼叫
        mock_opencc.assert_called_once_with("s2twp")
        mock_cc_instance.convert.assert_called_once_with(api_response)
        assert result == expected_traditional


class TestCorrectWordsOllama:
    """測試 correct_words_ollama 函數"""

    @patch("translator_ollama.requests.post")
    def test_correct_words_ollama_basic_correction(self, mock_post):
        """測試基本錯字修正功能"""
        # 準備測試資料 - 包含常見的同音異字錯誤
        input_text = "我們在這裡討論這個問題，的確需要在考慮一下。"
        expected_corrected = "我們在這裡討論這個問題，的確需要再考慮一下。"

        # 模擬 API 回應
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": expected_corrected}}
        mock_post.return_value = mock_response

        # 執行測試
        result = correct_words_ollama(input_text)

        # 驗證結果
        assert result == expected_corrected

        # 驗證 API 呼叫參數
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "gemma3:27b"
        assert call_args[1]["json"]["messages"][1]["content"] == input_text
        assert call_args[1]["json"]["stream"] is False

    @patch("translator_ollama.requests.post")
    def test_correct_words_ollama_whisper_common_errors(self, mock_post):
        """測試 Whisper 常見錯誤修正"""
        # 準備測試資料 - Whisper 常見的識別錯誤
        input_text = "今天我們要討論關於 AI 的發展，這個技術真的很厲害。"
        expected_corrected = "今天我們要討論關於 AI 的發展，這個技術真得很厲害。"

        # 模擬 API 回應
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": expected_corrected}}
        mock_post.return_value = mock_response

        # 執行測試
        result = correct_words_ollama(input_text)

        # 驗證結果
        assert result == expected_corrected

    @patch("translator_ollama.requests.post")
    def test_correct_words_ollama_preserve_formatting(self, mock_post):
        """測試保持原始格式（換行、空格、標點符號）"""
        # 準備測試資料 - 包含換行和特殊格式
        input_text = """第一段文字，有一些錯字。

第二段文字，也有一些問題。
    縮排的文字。

最後一段。"""

        expected_corrected = """第一段文字，有一些錯字。

第二段文字，也有一些問題。
    縮排的文字。

最後一段。"""

        # 模擬 API 回應
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": expected_corrected}}
        mock_post.return_value = mock_response

        # 執行測試
        result = correct_words_ollama(input_text)

        # 驗證結果保持格式
        assert result == expected_corrected

    @patch("translator_ollama.requests.post")
    def test_correct_words_ollama_system_prompt_validation(self, mock_post):
        """測試系統提示詞是否包含正確的指令"""
        # 準備測試資料
        input_text = "測試文字"
        expected_corrected = "測試文字"

        # 模擬 API 回應
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": expected_corrected}}
        mock_post.return_value = mock_response

        # 執行測試
        correct_words_ollama(input_text)

        # 驗證系統提示詞內容
        call_args = mock_post.call_args
        system_message = call_args[1]["json"]["messages"][0]["content"]

        # 檢查系統提示詞包含關鍵指令
        assert "文字校正專家" in system_message
        assert "Whisper" in system_message
        assert "錯字與同音異字" in system_message
        assert "只回傳校正後的文字" in system_message
        assert "嚴格遵守原始格式" in system_message
        assert "換行、空格、標點符號" in system_message
        assert "只將錯誤的字元替換為正確的字元" in system_message
        assert "絕對不要新增、刪除或改寫任何內容" in system_message

    @patch("translator_ollama.requests.post")
    def test_correct_words_ollama_custom_params(self, mock_post):
        """測試使用自定義參數"""
        # 準備測試資料
        input_text = "測試文字"
        custom_url = "http://custom-server:11434/api/chat"
        custom_model = "llama2:7b"
        expected_corrected = "測試文字"

        # 模擬 API 回應
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": expected_corrected}}
        mock_post.return_value = mock_response

        # 執行測試
        result = correct_words_ollama(input_text, url=custom_url, model=custom_model)

        # 驗證結果
        assert result == expected_corrected

        # 驗證 API 呼叫使用了自定義參數
        call_args = mock_post.call_args
        assert call_args[0][0] == custom_url
        assert call_args[1]["json"]["model"] == custom_model

    @patch("translator_ollama.requests.post")
    def test_correct_words_ollama_api_error(self, mock_post):
        """測試 API 錯誤處理"""
        # 準備測試資料
        input_text = "測試文字"

        # 模擬 API 錯誤
        mock_post.side_effect = requests.exceptions.RequestException("API 連線錯誤")

        # 執行測試並驗證例外
        with pytest.raises(requests.exceptions.RequestException):
            correct_words_ollama(input_text)

    @patch("translator_ollama.requests.post")
    def test_correct_words_ollama_empty_input(self, mock_post):
        """測試空輸入處理"""
        # 準備測試資料
        input_text = ""
        expected_corrected = ""

        # 模擬 API 回應
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": expected_corrected}}
        mock_post.return_value = mock_response

        # 執行測試
        result = correct_words_ollama(input_text)

        # 驗證結果
        assert result == expected_corrected

    def test_correct_words_ollama_real_world_examples(self):
        """測試真實世界的錯字修正範例（需要實際的 Ollama 服務）"""
        # 這個測試需要實際的 Ollama 服務運行，可以用來手動測試 prompt 效果
        test_cases = [
            {
                "input": "我們在討論這個問題的時候，的確需要在考慮一下。",
                "expected_keywords": ["再考慮"],  # 應該修正 "在考慮" 為 "再考慮"
            },
            {
                "input": "他說的話很有道理，我們應該採納他的意見。",
                "expected_keywords": ["得"],  # 可能需要修正 "的" 為 "得"
            },
            {
                "input": "這個方案比較好，我們可以先試試看。",
                "expected_keywords": ["比較"],  # 檢查是否保持正確
            },
        ]

        # 這個測試預設會被跳過，除非手動啟用
        pytest.skip("需要實際的 Ollama 服務運行，用於手動測試")

        for i, case in enumerate(test_cases):
            try:
                result = correct_words_ollama(case["input"])
                print(f"測試案例 {i + 1}:")
                print(f"輸入: {case['input']}")
                print(f"輸出: {result}")
                print(f"預期關鍵字: {case['expected_keywords']}")
                print("-" * 50)
            except Exception as e:
                pytest.fail(f"測試案例 {i + 1} 失敗: {e}")

    @patch("translator_ollama.requests.post")
    def test_correct_words_ollama_response_format(self, mock_post):
        """測試回應格式處理"""
        # 準備測試資料
        input_text = "測試文字"
        api_response_with_whitespace = "  修正後的文字  "
        expected_corrected = "修正後的文字"

        # 模擬 API 回應（包含前後空白）
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": api_response_with_whitespace}
        }
        mock_post.return_value = mock_response

        # 執行測試
        result = correct_words_ollama(input_text)

        # 驗證結果已去除前後空白
        assert result == expected_corrected


if __name__ == "__main__":
    pytest.main([__file__])
