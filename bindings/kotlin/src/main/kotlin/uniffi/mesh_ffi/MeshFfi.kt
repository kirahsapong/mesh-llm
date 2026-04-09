package uniffi.mesh_ffi

sealed class FfiException(message: String, cause: Throwable? = null) :
    kotlin.Exception(message, cause) {
    class InvalidInviteToken(message: String = "InvalidInviteToken", cause: Throwable? = null) :
        FfiException(message, cause)
    class JoinFailed(message: String = "JoinFailed", cause: Throwable? = null) :
        FfiException(message, cause)
    class DiscoveryFailed(message: String = "DiscoveryFailed", cause: Throwable? = null) :
        FfiException(message, cause)
    class StreamFailed(message: String = "StreamFailed", cause: Throwable? = null) :
        FfiException(message, cause)
    class Cancelled(message: String = "Cancelled", cause: Throwable? = null) :
        FfiException(message, cause)
    class ReconnectFailed(message: String = "ReconnectFailed", cause: Throwable? = null) :
        FfiException(message, cause)
}

data class ModelDto(
    val id: String,
    val name: String,
)

data class StatusDto(
    val connected: Boolean,
    val peerCount: ULong,
)

data class ChatMessageDto(
    val role: String,
    val content: String,
)

data class ChatRequestDto(
    val model: String,
    val messages: List<ChatMessageDto>,
)

data class ResponsesRequestDto(
    val model: String,
    val input: String,
)

sealed class EventDto {
    object Connecting : EventDto()
    data class Joined(val nodeId: String) : EventDto()
    data class ModelsUpdated(val models: List<ModelDto>) : EventDto()
    data class TokenDelta(val requestId: String, val delta: String) : EventDto()
    data class Completed(val requestId: String) : EventDto()
    data class Failed(val requestId: String, val error: String) : EventDto()
    data class Disconnected(val reason: String) : EventDto()
}

interface EventListener {
    fun onEvent(event: EventDto)
}

interface MeshClientHandleInterface {
    fun join()
    fun listModels(): List<ModelDto>
    fun chat(request: ChatRequestDto, listener: EventListener): String
    fun responses(request: ResponsesRequestDto, listener: EventListener): String
    fun cancel(requestId: String)
    fun status(): StatusDto
    fun disconnect()
    fun reconnect()
}
