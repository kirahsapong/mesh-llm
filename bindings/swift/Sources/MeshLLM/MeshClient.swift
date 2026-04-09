import Foundation

public struct InviteToken: Sendable {
    public let value: String

    public init(_ value: String) {
        self.value = value
    }
}

public struct Model: Sendable {
    public let id: String
    public let name: String
}

public struct MeshStatus: Sendable {
    public let connected: Bool
    public let peerCount: Int
}

public struct RequestId: Sendable {
    public let value: String
}

public enum MeshEvent: Sendable {
    case connecting
    case joined(nodeId: String)
    case modelsUpdated(models: [Model])
    case tokenDelta(requestId: String, delta: String)
    case completed(requestId: String)
    case failed(requestId: String, error: String)
    case disconnected(reason: String)
}

public struct ChatMessage: Sendable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public struct ChatRequest: Sendable {
    public let model: String
    public let messages: [ChatMessage]

    public init(model: String, messages: [ChatMessage]) {
        self.model = model
        self.messages = messages
    }
}

public struct ResponsesRequest: Sendable {
    public let model: String
    public let input: String

    public init(model: String, input: String) {
        self.model = model
        self.input = input
    }
}

public final class MeshClient: @unchecked Sendable {
    private let inviteToken: InviteToken
    private var isConnected: Bool = false
    private var eventHandlers: [(MeshEvent) -> Void] = []

    public init(inviteToken: InviteToken) {
        self.inviteToken = inviteToken
    }

    public func join() async throws {
        isConnected = true
        emit(.connecting)
        emit(.joined(nodeId: "local"))
    }

    public func listModels() async throws -> [Model] {
        return []
    }

    public func chat(_ request: ChatRequest) -> AsyncThrowingStream<MeshEvent, Error> {
        let requestId = UUID().uuidString
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { [weak self] _ in
                self?.cancel(RequestId(value: requestId))
            }
            continuation.yield(.completed(requestId: requestId))
            continuation.finish()
        }
    }

    public func responses(_ request: ResponsesRequest) -> AsyncThrowingStream<MeshEvent, Error> {
        let requestId = UUID().uuidString
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { [weak self] _ in
                self?.cancel(RequestId(value: requestId))
            }
            continuation.yield(.completed(requestId: requestId))
            continuation.finish()
        }
    }

    public func cancel(_ requestId: RequestId) {
    }

    public func status() async -> MeshStatus {
        return MeshStatus(connected: isConnected, peerCount: 0)
    }

    public func disconnect() async {
        isConnected = false
        emit(.disconnected(reason: "disconnect_requested"))
    }

    public func reconnect() async throws {
        await disconnect()
        try await join()
    }

    private func emit(_ event: MeshEvent) {
        for handler in eventHandlers {
            handler(event)
        }
    }
}
