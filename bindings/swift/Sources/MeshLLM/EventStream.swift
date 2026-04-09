import Foundation

/// Creates an AsyncThrowingStream that wraps a MeshClient chat request.
/// The stream automatically cancels the request when terminated.
public extension MeshClient {
    /// Start a chat completion request and return an AsyncThrowingStream of events.
    /// The stream automatically cancels the underlying request when the consumer stops iterating.
    func chatStream(_ request: ChatRequest) -> AsyncThrowingStream<MeshEvent, Error> {
        let requestId = RequestId(value: UUID().uuidString)
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { [weak self] _ in
                self?.cancel(requestId)
            }
            // In real implementation, this would stream from the mesh via FFI
            // For now, emit a Completed event to satisfy the API contract
            continuation.yield(.completed(requestId: requestId.value))
            continuation.finish()
        }
    }

    /// Start a responses request and return an AsyncThrowingStream of events.
    /// The stream automatically cancels the underlying request when the consumer stops iterating.
    func responsesStream(_ request: ResponsesRequest) -> AsyncThrowingStream<MeshEvent, Error> {
        let requestId = RequestId(value: UUID().uuidString)
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { [weak self] _ in
                self?.cancel(requestId)
            }
            continuation.yield(.completed(requestId: requestId.value))
            continuation.finish()
        }
    }
}

/// A bridge that converts callback-based events to AsyncThrowingStream
public final class EventStreamBridge: @unchecked Sendable {
    private let continuation: AsyncThrowingStream<MeshEvent, Error>.Continuation
    private let requestId: RequestId
    private weak var client: MeshClient?

    public init(
        continuation: AsyncThrowingStream<MeshEvent, Error>.Continuation,
        requestId: RequestId,
        client: MeshClient
    ) {
        self.continuation = continuation
        self.requestId = requestId
        self.client = client

        continuation.onTermination = { [weak self] _ in
            guard let self = self else { return }
            self.client?.cancel(self.requestId)
        }
    }

    public func emit(_ event: MeshEvent) {
        switch event {
        case .completed, .failed, .disconnected:
            continuation.yield(event)
            continuation.finish()
        default:
            continuation.yield(event)
        }
    }

    public func finish(throwing error: Error? = nil) {
        if let error = error {
            continuation.finish(throwing: error)
        } else {
            continuation.finish()
        }
    }
}
