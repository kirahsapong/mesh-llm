package ai.meshllm.smoke

import ai.meshllm.ChatMessage
import ai.meshllm.ChatRequest
import ai.meshllm.Event
import ai.meshllm.MeshClient
import kotlinx.coroutines.runBlocking
import uniffi.mesh_ffi.ChatRequestDto
import uniffi.mesh_ffi.EventDto
import uniffi.mesh_ffi.EventListener
import uniffi.mesh_ffi.MeshClientHandleInterface
import uniffi.mesh_ffi.ModelDto
import uniffi.mesh_ffi.ResponsesRequestDto
import uniffi.mesh_ffi.StatusDto
import java.util.UUID
import java.util.concurrent.CountDownLatch

private class MockMeshClientHandle : MeshClientHandleInterface {

    override fun join() {
        Thread.sleep(10)
    }

    override fun listModels(): List<ModelDto> = listOf(
        ModelDto(id = "mock-model-1", name = "Mock Model 1"),
        ModelDto(id = "mock-model-2", name = "Mock Model 2"),
        ModelDto(id = "mock-model-3", name = "Mock Model 3"),
    )

    override fun chat(request: ChatRequestDto, listener: EventListener): String {
        val requestId = UUID.randomUUID().toString()
        Thread {
            Thread.sleep(5)
            listener.onEvent(EventDto.TokenDelta(requestId = requestId, delta = "Hello"))
            Thread.sleep(5)
            listener.onEvent(EventDto.TokenDelta(requestId = requestId, delta = " world"))
            Thread.sleep(5)
            listener.onEvent(EventDto.Completed(requestId = requestId))
        }.also { it.isDaemon = true }.start()
        return requestId
    }

    override fun responses(request: ResponsesRequestDto, listener: EventListener): String {
        val requestId = UUID.randomUUID().toString()
        Thread {
            listener.onEvent(EventDto.Completed(requestId = requestId))
        }.also { it.isDaemon = true }.start()
        return requestId
    }

    override fun cancel(requestId: String) = Unit

    override fun status(): StatusDto = StatusDto(connected = true, peerCount = 1UL)

    override fun disconnect() {
        Thread.sleep(5)
    }

    override fun reconnect() = Unit
}

fun main(args: Array<String>) = runBlocking {
    val isMock = args.contains("--mock")
    val inviteToken = args.firstOrNull { !it.startsWith("--") }

    if (!isMock && inviteToken == null) {
        System.err.println("Usage: SmokeMain <invite_token>")
        System.err.println("       SmokeMain --mock")
        System.exit(1)
    }

    val handle: MeshClientHandleInterface = if (isMock) {
        MockMeshClientHandle()
    } else {
        throw UnsupportedOperationException(
            "Real mode requires the native mesh-ffi library on java.library.path. " +
                "Build with `cargo build --release -p mesh-ffi`, then pass your invite token as the first argument."
        )
    }

    val client = MeshClient(handle)

    client.join()
    println("[connected]")

    val models = client.listModels()
    println("[models] N=${models.size}")

    val chatRequest = ChatRequest(
        model = models.firstOrNull()?.id ?: "mock-model-1",
        messages = listOf(ChatMessage(role = "user", content = "hello")),
    )

    val latch = CountDownLatch(1)
    var firstTokenEmitted = false
    val chatStartMs = System.currentTimeMillis()

    client.chat(chatRequest) { event ->
        when (event) {
            is Event.TokenDelta -> {
                if (!firstTokenEmitted) {
                    firstTokenEmitted = true
                    val elapsedMs = System.currentTimeMillis() - chatStartMs
                    println("[chat] first_token_ms=$elapsedMs")
                }
            }
            is Event.Completed -> latch.countDown()
            is Event.Failed -> {
                System.err.println("[chat] failed: ${event.error}")
                latch.countDown()
            }
            else -> Unit
        }
    }

    latch.await()
    println("[chat] done")

    client.disconnect()
    println("[disconnect] ok")
}
