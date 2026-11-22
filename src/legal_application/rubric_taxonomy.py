"""Category mappings and judgment-analysis groups used during rubric construction."""

TAXONOMY = {
    "Criminal": {
        "procedure": "Tính hợp pháp của hoạt động, quyết định tố tụng và của VKS",
        "offense": "Hành vi, chứng cứ, dấu hiệu cấu thành và tội danh",
        "severity": "Tính chất và mức độ nguy hiểm của hành vi",
        "sentencing_factors": "Tăng nặng, giảm nhẹ và căn cứ quyết định hình phạt",
        "sentence": "Hình phạt, miễn trách nhiệm hoặc hình phạt, biện pháp tư pháp",
        "civil_liability": "Trách nhiệm dân sự",
        "property": "Xử lý vật chứng, tài sản",
        "fees_appeal": "Án phí, quyền kháng cáo và vấn đề liên quan",
    },
    "Civil": {
        "relationship": "Quan hệ pháp luật tranh chấp",
        "procedure": "Thẩm quyền và thủ tục tố tụng",
        "claims": "Yêu cầu khởi kiện, phản tố và yêu cầu độc lập",
        "evidence": "Chứng cứ và tình tiết được xác định",
        "obligations": "Quyền, nghĩa vụ, hiệu lực giao dịch, thực hiện hoặc vi phạm nghĩa vụ",
        "disposition": "Căn cứ pháp luật, chấp nhận hoặc bác yêu cầu và hậu quả pháp lý",
        "fees_appeal": "Án phí, chi phí tố tụng và kháng cáo",
    },
    "Commercial": {
        "relationship": "Quan hệ kinh doanh, thương mại và tư cách của các bên",
        "procedure": "Thẩm quyền, thủ tục và điều kiện giải quyết tranh chấp",
        "claims": "Yêu cầu khởi kiện, phản tố và yêu cầu độc lập",
        "evidence": "Chứng cứ, giao dịch và tình tiết được Tòa xác định",
        "contract": "Giao kết, hiệu lực và nội dung của giao dịch thương mại",
        "obligations": "Thực hiện, vi phạm nghĩa vụ và căn cứ miễn trách nhiệm",
        "remedies": "Thanh toán, lãi, phạt vi phạm, bồi thường và hậu quả pháp lý",
        "disposition": "Căn cứ chấp nhận hoặc bác yêu cầu",
        "fees_appeal": "Án phí, chi phí tố tụng và kháng cáo",
    },
    "Marriage & Family": {
        "relationship": "Quan hệ hôn nhân gia đình và tính hợp pháp của hôn nhân",
        "marriage": "Căn cứ giải quyết yêu cầu về hôn nhân",
        "children": "Con chung, quyền nuôi con và cấp dưỡng",
        "property": "Tài sản chung và tài sản riêng",
        "debts": "Nợ chung và nghĩa vụ tài sản",
        "procedure": "Thẩm quyền và thủ tục tố tụng",
        "fees_appeal": "Án phí và kháng cáo",
    },
    "Labor": {
        "relationship": "Quan hệ pháp luật lao động và chủ thể",
        "contract": "Điều kiện và hiệu lực hợp đồng lao động",
        "obligations": "Quyền và nghĩa vụ của các bên",
        "legality": "Tính hợp pháp của hành vi hoặc quyết định bị tranh chấp",
        "remedies": "Tiền lương, bảo hiểm, trợ cấp, bồi thường và hậu quả pháp lý",
        "procedure": "Điều kiện khởi kiện, hòa giải, thời hiệu và thủ tục tố tụng",
        "fees_appeal": "Án phí và kháng cáo",
    },
    "Administrative": {
        "challenged_act": "Đối tượng khởi kiện",
        "procedure": "Thời hiệu, thẩm quyền và thủ tục tố tụng",
        "legality": "Tính hợp pháp của quyết định hoặc hành vi hành chính",
        "authority": "Thẩm quyền ban hành quyết định hoặc thực hiện hành vi",
        "requirements": "Trình tự, thủ tục, hình thức, thời hạn và nội dung",
        "effects": "Quyền, lợi ích bị ảnh hưởng và hậu quả pháp lý",
        "damages": "Bồi thường thiệt hại",
        "fees_appeal": "Án phí và kháng cáo",
    },
}

CATEGORY_ALIASES = {
    **{name.casefold(): name for name in TAXONOMY},
    "hình sự": "Criminal", "dân sự": "Civil",
    "kinh doanh thương mại": "Commercial", "kinh doanh, thương mại": "Commercial",
    "marriage and family": "Marriage & Family", "hôn nhân và gia đình": "Marriage & Family",
    "hngđ": "Marriage & Family", "labour": "Labor", "lao động": "Labor", "hành chính": "Administrative",
}

def canonical_category(value: str) -> str:
    if not isinstance(value, str) or value.strip().casefold() not in CATEGORY_ALIASES:
        raise ValueError(f"Unsupported category {value!r}")
    return CATEGORY_ALIASES[value.strip().casefold()]
