# 文件服务接入

所有上传、URL 导入副本和处理结果均保存到 file-service。成功响应保留稳定 UUID `file_id` 和 `files[]`；`file_url` 是短期链接，过期后按 `file_id` 重新获取。业务文档、任务 ID 与文件服务 ID 分开使用。

服务端配置：

```dotenv
FILE_SERVICE_BASE_URL=https://game.toolup.cn
FILE_SERVICE_TOKEN=
FILE_SERVICE_STORAGE_NAMESPACE=default
FILE_SERVICE_TIMEOUT_SECONDS=120
```

根地址不带 `/api`。Token 使用文件服务认可且具有对应权限的服务身份凭据；留空只适用于服务明确允许匿名访问时。多个应用可以共享同一身份，但也将共享其权限边界。namespace 是文件服务配置允许的存储标签，不是 bucket 或权限隔离机制；`default` 必须在文件服务配置中存在。浏览器不能持有服务身份 Token。

未配置或文件服务失败时返回错误，不降级为本地下载。普通 HTTP(S) URL（包括原有对象存储链接）经大小、超时、重定向和网络地址检查后读取，再上传独立副本获得新 `file_id`。源链接本身不作为新文件身份。

上传不自动重试；上传已完成但链接签发失败时，使用错误携带的 `file_id` 恢复。删除临时工作文件不会删除文件服务文件。下载重新检查当前访问权限，过期或撤销不能通过业务下载接口绕过。

修改配置后需自行重新启动/部署相关服务；本次迁移未部署，也未删除已有远端文件。
