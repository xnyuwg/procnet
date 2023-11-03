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


class PseudoDocEELabel:
    EVENT_SCHEMA = {'解除质押': ['质押方', '披露时间', '质权方', '质押物', '质押股票/股份数量', '事件时间', '质押物所属公司', '质押物占总股比', '质押物占持股比'], '股份回购': ['回购方', '披露时间', '回购股份数量', '每股交易价格', '占公司总股本比例', '交易金额', '回购完成时间'],
     '股东减持': ['股票简称', '披露时间', '交易股票/股份数量', '每股交易价格', '交易金额', '交易完成时间', '减持方', '减持部分占所持比例', '减持部分占总股本比例'], '亏损': ['公司名称', '披露时间', '财报周期', '净亏损', '亏损变化'],
     '中标': ['中标公司', '中标标的', '中标金额', '招标方', '中标日期', '披露日期'], '高管变动': ['高管姓名', '任职公司', '高管职位', '事件时间', '变动类型', '披露日期', '变动后职位', '变动后公司名称'], '企业破产': ['破产公司', '披露时间', '债务规模', '破产时间', '债权人'],
     '股东增持': ['股票简称', '披露时间', '交易股票/股份数量', '每股交易价格', '交易金额', '交易完成时间', '增持方', '增持部分占所持比例', '增持部分占总股本比例'], '被约谈': ['公司名称', '披露时间', '被约谈时间', '约谈机构'],
     '企业收购': ['收购方', '披露时间', '被收购方', '收购标的', '交易金额', '收购完成时间'], '公司上市': ['上市公司', '证券代码', '环节', '披露时间', '发行价格', '事件时间', '市值', '募资金额'],
     '企业融资': ['投资方', '披露时间', '被投资方', '融资金额', '融资轮次', '事件时间', '领投方'], '质押': ['质押方', '披露时间', '质权方', '质押物', '质押股票/股份数量', '事件时间', '质押物占总股比', '质押物所属公司', '质押物占持股比']}
    EVENT_TYPE = EVENT_SCHEMA.keys()
    KEY_ENG_CHN = {"EventType": "事件", "Null": "无"}
    for k, vs in EVENT_SCHEMA.items():
        for v in vs:
            KEY_ENG_CHN[v] = v
    KEY_CHN_ENG = {v: k for k, v in KEY_ENG_CHN.items()}


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
