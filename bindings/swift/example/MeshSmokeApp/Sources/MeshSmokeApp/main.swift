import Foundation
import MeshLLM

let args = Array(CommandLine.arguments.dropFirst())
let isMock = args.contains("--mock")
let inviteTokenArg = args.first { !$0.hasPrefix("--") }

if isMock {
    print("[connected]")
    print("[models] N=3")
    let chatStartMs = Int(Date().timeIntervalSince1970 * 1000)
    Thread.sleep(forTimeInterval: 0.005)
    let firstTokenMs = Int(Date().timeIntervalSince1970 * 1000) - chatStartMs
    print("[chat] first_token_ms=\(firstTokenMs)")
    print("Hello world")
    print("[chat] done")
    print("[disconnect] ok")
    exit(0)
}

let token = inviteTokenArg ?? "smoke-test-token"
let client = MeshClient(inviteToken: InviteToken(token))

print("[connected]")

Task {
    do {
        let models = try await client.listModels()
        print("[models] N=\(models.count)")
        
        let request = ChatRequest(
            model: models.first?.id ?? "default",
            messages: [ChatMessage(role: "user", content: "hello")]
        )
        
        let startTime = Date()
        var firstToken = true
        for try await event in client.chatStream(request) {
            switch event {
            case .tokenDelta(_, let delta):
                if firstToken {
                    let ms = Int(Date().timeIntervalSince(startTime) * 1000)
                    print("[chat] first_token_ms=\(ms)")
                    firstToken = false
                }
                print(delta, terminator: "")
            case .completed:
                print("\n[chat] done")
            case .failed(_, let error):
                print("[chat] error: \(error)")
            default:
                break
            }
        }
        
        await client.disconnect()
        print("[disconnect] ok")
    } catch {
        print("[error] \(error)")
    }
    exit(0)
}

RunLoop.main.run()
