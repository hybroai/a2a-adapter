# Getting Started with Local Debugging 🚀

> Step-by-step guide to debug A2A Adapters SDK locally

## 🎯 Goal

Complete environment setup and run your first debug test in 5 minutes.

## 📋 Prerequisites

- Python 3.9 or higher
- Terminal/command line access
- (Optional) VS Code or PyCharm

## 🚀 Step 1: Environment Setup

### Method 1: Automatic Setup (Recommended)

```bash
cd "/Users/caijiangnan/Desktop/HYBRO AI/multiple-agents/hybro open source/a2a-adapter"

# Run setup script
./setup_dev.sh
```

This script will automatically:
- ✅ Check Python version
- ✅ Create virtual environment
- ✅ Install development dependencies
- ✅ Run validation tests

### Method 2: Manual Setup

```bash
cd "/Users/caijiangnan/Desktop/HYBRO AI/multiple-agents/hybro open source/a2a-adapter"

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import a2a_adapter; print('✅ Installation successful!')"
```

## 🧪 第二步：运行第一个测试

```bash
# 确保虚拟环境已激活
source venv/bin/activate

# 运行最简单的测试
python debug_scripts/01_simple_test.py
```

**期望输出：**
```
============================================================
🧪 A2A Adapters - 简单本地测试
============================================================

📦 步骤 1: 加载 callable adapter...
✅ Adapter 加载成功

📝 步骤 2: 创建测试消息...
✅ 测试消息创建成功

🚀 步骤 3: 调用 adapter.handle()...
✅ Agent 收到消息: hello world
✅ 调用成功

📊 步骤 4: 结果分析
   - 角色: assistant
   - 内容类型: <class 'list'>
   - 内容: Echo: HELLO WORLD

============================================================
✅ 测试完成！所有功能正常工作
============================================================
```

如果看到这个输出，恭喜！环境设置成功 🎉

## 🔍 第三步：理解数据流

运行带日志的测试，查看详细的执行流程：

```bash
python debug_scripts/02_debug_with_logging.py
```

这会显示：
- Adapter 如何加载
- 消息如何转换
- 每个步骤的输入输出
- 最终结果的生成

## 🌐 第四步：测试完整服务器

### 终端 1: 启动服务器

```bash
source venv/bin/activate
python examples/05_custom_adapter.py
```

选择 `1` (Custom BaseAgentAdapter subclass)

### 终端 2: 测试服务器

```bash
source venv/bin/activate
python debug_scripts/03_test_server_client.py
```

这会测试运行中的服务器并显示响应。

## 🎨 第五步：VS Code 可视化调试（推荐）

### 1. 打开项目

```bash
code "/Users/caijiangnan/Desktop/HYBRO AI/multiple-agents/hybro open source/a2a-adapter"
```

### 2. 设置断点

在 `a2a_adapter/adapter.py` 文件中，点击行号左侧设置断点：

```python
async def handle(self, params: MessageSendParams) -> Message | Task:
    framework_input = await self.to_framework(params)  # ← 在这里设置断点
    framework_output = await self.call_framework(framework_input, params)
    return await self.from_framework(framework_output, params)
```

### 3. 开始调试

- 按 `F5`
- 选择 "🧪 调试: 简单测试"
- 程序会在断点处暂停

### 4. 调试操作

- **F10** - 单步跳过（Step Over）
- **F11** - 单步进入（Step Into）
- **Shift+F11** - 单步跳出（Step Out）
- **F5** - 继续（Continue）
- **查看变量** - 鼠标悬停在变量上查看值

## 🛠️ 常见调试场景

### 场景 1: 调试自定义 Agent

创建测试文件 `my_debug.py`：

```python
import asyncio
from a2a_adapter import load_a2a_agent
from a2a.types import MessageSendParams, Message, TextPart

async def my_agent(inputs: dict) -> str:
    # 在这里添加你的调试代码
    print(f"[DEBUG] 收到输入: {inputs}")
    
    message = inputs["message"]
    
    # 你的处理逻辑
    result = f"处理: {message.upper()}"
    
    print(f"[DEBUG] 返回结果: {result}")
    return result

async def main():
    adapter = await load_a2a_agent({
        "adapter": "callable",
        "callable": my_agent
    })
    
    params = MessageSendParams(
        messages=[
            Message(
                role="user",
                content=[TextPart(type="text", text="测试消息")]
            )
        ]
    )
    
    result = await adapter.handle(params)
    print(f"最终结果: {result.content[0].text}")

if __name__ == "__main__":
    asyncio.run(main())
```

运行：

```bash
python my_debug.py
```

### 场景 2: 调试 N8n Adapter

```python
import asyncio
import logging
from a2a_adapter.integrations.n8n import N8nAgentAdapter
from a2a.types import MessageSendParams, Message, TextPart

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

async def test_n8n():
    adapter = N8nAgentAdapter(
        webhook_url="https://webhook.site/your-unique-url",  # 使用 webhook.site 测试
        timeout=10
    )
    
    params = MessageSendParams(
        messages=[
            Message(
                role="user",
                content=[TextPart(type="text", text="test")]
            )
        ]
    )
    
    try:
        result = await adapter.handle(params)
        print(f"✅ 成功: {result.content[0].text}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await adapter.close()

if __name__ == "__main__":
    asyncio.run(test_n8n())
```

### 场景 3: 性能分析

```bash
python debug_scripts/04_benchmark.py
```

这会显示：
- 平均响应时间
- 吞吐量（请求/秒）
- 性能瓶颈

## 📝 调试检查清单

遇到问题时，按顺序检查：

- [ ] 虚拟环境已激活？`which python`
- [ ] 包已安装？`pip list | grep a2a-adapter`
- [ ] Python 版本正确？`python --version` (>= 3.9)
- [ ] 能导入包？`python -c "import a2a_adapter"`
- [ ] 端口未被占用？`lsof -i :9000`
- [ ] 环境变量已设置？（如 OPENAI_API_KEY）

## 🆘 故障排除

### 错误: ModuleNotFoundError

```bash
# 解决方案
pip install -e .
```

### 错误: Address already in use

```bash
# 查找占用端口的进程
lsof -i :9000

# 杀死进程
kill -9 <PID>
```

### 错误: Permission denied

```bash
# 确保脚本有执行权限
chmod +x setup_dev.sh
```

### 错误: 虚拟环境问题

```bash
# 删除并重建虚拟环境
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## 📚 下一步

现在你已经掌握了基础调试，可以：

1. **查看示例** - 浏览 `examples/` 目录
2. **阅读详细文档** - [LOCAL_DEVELOPMENT.md](LOCAL_DEVELOPMENT.md)
3. **快速参考** - [DEBUG_QUICKREF.md](DEBUG_QUICKREF.md)
4. **创建自定义 Adapter** - 参考 `examples/05_custom_adapter.py`
5. **贡献代码** - 阅读 [CONTRIBUTING.md](CONTRIBUTING.md)

## 💡 调试技巧

1. **从简单开始** - 先运行最简单的测试
2. **逐步深入** - 一次只调试一个组件
3. **查看日志** - 使用详细日志了解执行流程
4. **使用断点** - VS Code 断点调试最直观
5. **写测试** - 为每个功能写测试用例

## 🎓 学习资源

- 📖 [README.md](README.md) - 完整文档
- 🚀 [QUICKSTART.md](QUICKSTART.md) - 快速开始
- 🏗️ [ARCHITECTURE.md](ARCHITECTURE.md) - 架构设计
- 💻 [examples/](examples/) - 实际示例
- 🧪 [tests/](tests/) - 测试用例

## 🎉 恭喜！

你已经完成了本地调试入门。现在你可以：

- ✅ 设置开发环境
- ✅ 运行和调试测试
- ✅ 启动和测试服务器
- ✅ 使用 VS Code 可视化调试
- ✅ 创建自定义 agent

准备好构建你的第一个 A2A agent 了吗？ 🚀
