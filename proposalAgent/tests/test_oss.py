# -*- coding: utf-8 -*-
"""上传测试文件到 OSS 并生成可下载的预签名 URL。"""

import alibabacloud_oss_v2 as oss
import oss2
from alibabacloud_oss_v2.credentials import EnvironmentVariableCredentialsProvider as V2EnvProvider
from oss2.credentials import EnvironmentVariableCredentialsProvider


# 从环境变量中获取访问凭证。运行本代码示例之前，请确保已设置
# OSS_ACCESS_KEY_ID 和 OSS_ACCESS_KEY_SECRET。
oss2_credentials_provider = EnvironmentVariableCredentialsProvider()
auth = oss2.ProviderAuthV4(oss2_credentials_provider)


# 基础配置，与原有上传示例保持一致
endpoint = "https://oss-cn-hangzhou.aliyuncs.com"
region = "cn-hangzhou"
bucket_name = "evaluatoin-pdfs"
object_key = "exampleobject.pdf"


def upload_text() -> None:
    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)

    pdf_content = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R>>endobj\n4 0 obj<</Length 44>>stream\nBT/F1 24 Tf 72 120 Td (Hello OSS PDF) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000060 00000 n\n0000000114 00000 n\n0000000195 00000 n\ntrailer<</Size 5/Root 1 0 R>>\nstartxref\n271\n%%EOF"

    headers = {"Content-Type": "application/pdf"}
    result = bucket.put_object(object_key, pdf_content, headers=headers)

    print("http status: {0}".format(result.status))
    print("request_id: {0}".format(result.request_id))
    print("ETag: {0}".format(result.etag))
    print("date: {0}".format(result.headers["date"]))


def generate_presigned_url() -> None:
    # 使用新版 SDK 生成 GET 预签名 URL
    cfg = oss.config.load_default()
    cfg.credentials_provider = V2EnvProvider()
    cfg.region = region
    cfg.endpoint = endpoint

    client = oss.Client(cfg)
    pre_result = client.presign(
        oss.GetObjectRequest(
            bucket=bucket_name,
            key=object_key,
        )
    )

    print(
        "method: {method}, expiration: {expiration}, url: {url}".format(
            method=pre_result.method,
            expiration=pre_result.expiration.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            url=pre_result.url,
        )
    )

    for header_key, header_value in pre_result.signed_headers.items():
        print(
            "signed headers key: {0}, signed headers value: {1}".format(
                header_key, header_value
            )
        )


if __name__ == "__main__":
    upload_text()
    generate_presigned_url()