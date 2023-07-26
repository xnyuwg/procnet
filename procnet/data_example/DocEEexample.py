from dataclasses import dataclass
from typing import List, Dict
import copy


class DocEELabel:
    EVENT_TYPE = ["EquityFreeze", "EquityRepurchase", "EquityUnderweight", "EquityOverweight", "EquityPledge"]
    KEY_ENG_CHN = {"EventType": "事件",
                   "EquityFreeze": "股权冻结",
                   "EquityRepurchase": "股权回购",
                   "EquityUnderweight": "股票减持",
                   "EquityOverweight": "股票增持",
                   "EquityPledge": "股权质押",
                   "Pledger": "质押者",
                   "PledgedShares": "质押股份",
                   "Pledgee": "质权人",
                   "TotalHoldingShares": "总持股",
                   "TotalHoldingRatio": "总持股比率",
                   "TotalPledgedShares": "总质押股份",
                   "StartDate": "开始日期",
                   "EndDate": "结束日期",
                   "ReleasedDate": "发布日期",
                   "EquityHolder": "股权持有人",
                   "TradedShares": "交易股票",
                   "LaterHoldingShares": "后来控股股份",
                   "AveragePrice": "平均价格",
                   "CompanyName": "公司名",
                   "HighestTradingPrice": "最高交易价",
                   "LowestTradingPrice": "最低交易价",
                   "RepurchasedShares": "回购股份",
                   "ClosingDate": "截止日期",
                   "RepurchaseAmount": "回购金额",
                   "FrozeShares": "冻结股份",
                   "LegalInstitution": "法律机构",
                   "UnfrozeDate": "解冻日期",
                   "Null": "无"
                   }
    KEY_CHN_ENG = {v: k for k, v in KEY_ENG_CHN.items()}
    EVENT_SCHEMA = {
        "EquityFreeze": [
            "EquityHolder",
            "FrozeShares",
            "LegalInstitution",
            "TotalHoldingShares",
            "TotalHoldingRatio",
            "StartDate",
            "EndDate",
            "UnfrozeDate",
        ],
        "EquityRepurchase": [
            "CompanyName",
            "HighestTradingPrice",
            "LowestTradingPrice",
            "RepurchasedShares",
            "ClosingDate",
            "RepurchaseAmount",
        ],
        "EquityUnderweight": [
            "EquityHolder",
            "TradedShares",
            "StartDate",
            "EndDate",
            "LaterHoldingShares",
            "AveragePrice",
        ],
        "EquityOverweight": [
            "EquityHolder",
            "TradedShares",
            "StartDate",
            "EndDate",
            "LaterHoldingShares",
            "AveragePrice",
        ],
        "EquityPledge": [
            "Pledger",
            "PledgedShares",
            "Pledgee",
            "TotalHoldingShares",
            "TotalHoldingRatio",
            "TotalPledgedShares",
            "StartDate",
            "EndDate",
            "ReleasedDate",
        ],
    }


@dataclass
class DocEEEntity:
    span: str
    positions: List[list]  # [[sentence_index, start, end]]
    field: str


@dataclass
class DocEEDocumentExample:
    doc_id: str
    sentences: List[str]
    entities: List[DocEEEntity]
    events: List[Dict[str, str]]  # must have a key named eventType
    sentences_token: List[List[str]] = None
    seq_BIO_tags: List[List[str]] = None

    def copy(self):
        return copy.deepcopy(self)

    def get_fragment(self, start_sen: int, end_sen: int):
        new_example = self.copy()
        new_example.sentences = self.sentences[start_sen: end_sen]
        new_example.sentences_token = self.sentences_token[start_sen: end_sen] if self.sentences_token is not None else None
        new_example.seq_BIO_tags = self.seq_BIO_tags[start_sen: end_sen] if self.seq_BIO_tags is not None else None
        return new_example
